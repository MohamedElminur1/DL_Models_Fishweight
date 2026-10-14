"""Generate Table 7 in long form so exact saved hyperparameters stay readable."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (MISSING, STOP, TASK_NAMES, SAMPLE_SET_NAMES, group_name, literal, model_name, models, setup, stops, stop_notes, write_table)


def main() -> None:
    sources,cfg=setup()
    stopped=stops(sources)
    rows=[]
    for task in ('T1','T2'):
        entries=[(g,m,'eligible') for g,m in models(cfg,task)] + [('all_sample_set',m,'all') for m in cfg['models'][task]['all_sample_set']]
        for group,model,sample_set in entries:
            prefix=[TASK_NAMES[task],SAMPLE_SET_NAMES[sample_set],group_name(group),model_name(model)]
            if (task,model) in stopped:
                rows.append(prefix+[STOP]+[MISSING]*5)
                continue
            suffix='__all' if sample_set=='all' else ''
            saved=sources.csv(f'results/tuning/{task}/lopo/{model}{suffix}/chosen.csv')
            for r in sorted(saved,key=lambda r:int(r['outer_fold'])):
                parameters=[k for k in r if k.startswith('hp_') and r[k]!='']
                fields=[r['outer_fold'],r['test_ponds'],r['best_trial'] or MISSING,r['epochs'] or MISSING]
                for key in parameters or [None]:
                    rows.append(prefix+fields+[literal(key.removeprefix('hp_')) if key else MISSING, literal(r[key]) if key else MISSING])
    write_table('chosen_hyperparameters',['Task','Sample set','Group','Model','Outer fold','Held-out pond','Chosen trial','Final-fit epochs','Parameter','Chosen value'],rows,sources,[
        'Title: Chosen LOPO hyperparameters per model and outer fold. One row per parameter; baselines without tuned parameters retain one row per fold with –. Values preserve source precision. Trial indices are zero-based.',
        'Eligible and descriptive all-events fits are kept separate. Deep final-fit epochs are the saved chosen-trial median rule. Stopped runs have no complete chosen.csv; – does not invent a fold or trial. Calendar fits are outside this LOPO table.',*stop_notes(stopped),
    ])

if __name__ == '__main__':
    main()
