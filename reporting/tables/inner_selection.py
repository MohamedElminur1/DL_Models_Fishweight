"""Generate Table 6 from inner scores, including unavailable stopped runs."""
from collections import defaultdict
import statistics
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (MISSING, STOP, TASK_NAMES, group_name, model_name, models, number, selections, setup, stops, stop_notes, write_table)


def main() -> None:
    sources,cfg = setup()
    primary = selections(sources)
    stopped = stops(sources)
    rows=[]
    for task in ('T1','T2'):
        for group,model in models(cfg,task):
            identity=[TASK_NAMES[task],group_name(group),model_name(model)]
            if (task,model) in stopped:
                rows.append(identity+[MISSING,STOP,'Unavailable: stopped (plan 9.0)'])
                continue
            saved=sources.csv(f'results/tuning/{task}/lopo/{model}/inner_scores.csv')
            folds=defaultdict(list)
            for r in saved:
                folds[r['outer_fold']].append(float(r['S']))
            if set(map(int,folds)) != set(primary[task]['outer_folds']) or any(len(v)!=cfg['design']['n_inner_folds'] for v in folds.values()):
                raise ValueError('incomplete inner scores')
            mean=statistics.mean(statistics.mean(v) for v in folds.values())
            selected='Primary baseline' if model==primary[task]['primary_baseline'] else 'Primary deep model' if model==primary[task]['primary_deep'] else 'Not selected' if group in ('time_only','deep') else 'Outside primary selection pools'
            rows.append(identity+[str(len(folds)),number(mean,'s'),selected])
    write_table('inner_selection',['Task','Group','Model','Outer folds','Mean inner-validation S','Selection'],rows,sources,[
        'Title: Inner-validation scores and primary model selection, plan 9.4. Mean S averages equally over inner folds within each outer fold, then equally over outer folds. Tuned models use their chosen trial; T2 S averages the two horizons.',
        'Only baseline and deep pools compete; ties follow frozen model order. Choices are read from committed selection JSON files, never from outer errors. Stops have no complete inner-score file.', *stop_notes(stopped),
    ])

if __name__ == '__main__':
    main()
