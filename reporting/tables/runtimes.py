"""Generate Table 12 with summed logged stage times and recorded hardware."""
from collections import defaultdict
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (MISSING, STOP, TASK_NAMES, SCHEME_NAMES, SAMPLE_SET_NAMES, literal, model_name, setup, stops, stop_notes, write_table)


def main() -> None:
    sources,cfg=setup()
    saved=sources.csv('results/logs/runtime.csv')
    stopped=stops(sources)
    groups=defaultdict(list)
    for r in saved:
        if r['scheme']!='permutation':
            groups[(r['task'],r['scheme'],r['model'])].append(r)
    rows=[]
    shown_stops=set()
    for (task,scheme,code),part in sorted(groups.items()):
        model=code.removesuffix('__all')
        times=defaultdict(list)
        for r in part:
            times[r['stage']].append(float(r['seconds']))
        perm=[r for r in saved if r['task']==task and r['model']==code and r['stage']=='permutation'] if scheme=='lopo' and not code.endswith('__all') else []
        hardware={(r['cpu'],r['python'],r['torch']) for r in part+perm}
        if len(hardware)!=1:
            raise ValueError('inconsistent hardware in one run')
        cpu,python,torch=hardware.pop()
        is_stopped=(task,model) in stopped and scheme=='lopo'
        if is_stopped:
            shown_stops.add((task,model))
        status=MISSING if is_stopped else 'Complete'
        seconds=lambda stage:f"{math.fsum(times[stage]):.1f}" if times[stage] else MISSING
        rows.append([TASK_NAMES[task],SCHEME_NAMES[scheme],SAMPLE_SET_NAMES['all' if code.endswith('__all') else 'eligible'],model_name(model),
                     STOP if is_stopped else seconds('trial'),MISSING if is_stopped else seconds('final'),
                     MISSING if is_stopped else f"{math.fsum(float(r['seconds']) for r in perm):.1f}" if perm else MISSING,
                     cpu,python,literal(torch),status])
    write_table('runtimes',['Task','Evaluation','Sample set','Model','Tuning time (s)','Final-fit time (s)','Permutation time (s)','CPU','Python','PyTorch','Status'],rows,sources,[
        'Title: Logged wall times and hardware, plan 11. Tuning is the sum of trial-stage seconds, including baseline inner fits; final fits sum all folds and seeds. Permutation totals attach to the eligible LOPO row only. – means no such recorded stage.',
        'Stage sums are elapsed times of logged fits, not end-to-end pipeline duration; parallel workers can overlap. Plan/check/skip/error records are not fit times. Skipped completed folds are not counted again. The stopped runs have no logged trial or final-fit durations; the first time cell shows stopped (plan 9.0), and the other time cells are –.',
        'Hardware and software versions are copied from runtime.csv. No GPU, RAM, thread count or operating-system field is saved in that file; PyTorch reports a CPU build.',*stop_notes(shown_stops),
    ])

if __name__ == '__main__':
    main()
