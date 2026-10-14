"""Generate Table 11, pond-level metrics across eligible and all-event evaluations."""
import statistics
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (SCHEME_NAMES, SAMPLE_SET_NAMES, group_name, horizon_name, metric, model_name, models, number, select, setup, stops, stop_notes, write_table)


def main() -> None:
    sources,cfg=setup()
    saved=sources.csv('results/metrics/per_pond.csv')
    seeded=sources.csv('results/metrics/per_pond_seed.csv')
    stopped=stops(sources)
    rows=[]
    notes=[]
    for r in saved:
        if r['horizon']=='all':
            continue
        model=r['model']
        group=next(g for g,m in models(cfg,r['task']) if m==model)
        if r['sample_set']=='all':
            group='all_sample_set'
        values=[]
        for key,kind in [('S','s'),('mae_g','grams'),('mape_pct','percent')]:
            if int(r['n_seeds'])>1:
                criteria={k:r[k] for k in ('task','scheme','sample_set','horizon','model','pond')}
                per_seed=select(seeded,**criteria)
                if {int(p['seed']) for p in per_seed} != set(cfg['seeds']['final']):
                    raise ValueError('incomplete pond seeds')
                copy=r|{key+'_sd':str(statistics.stdev(float(p[key]) for p in per_seed))}
                values.append(metric(copy,key,kind))
            else:
                values.append(number(r[key],kind))
        rows.append([horizon_name(r['task'],int(r['horizon'])),SCHEME_NAMES[r['scheme']],SAMPLE_SET_NAMES[r['sample_set']],
                     group_name(group),model_name(model),r['pond'],r['n'],r['n_seeds'],*values])
        if float(r['n_nonpositive'])>0:
            notes.append(f"{horizon_name(r['task'],int(r['horizon']))}, {SCHEME_NAMES[r['scheme']]}, {model_name(model)}, pond {r['pond']}: {int(float(r['n_nonpositive']))} non-positive predictions.")
    write_table('per_pond',['Task / horizon','Evaluation','Sample set','Group','Model','Pond','Samples','Seeds','S','MAE (g)','MAPE (%)'],rows,sources,[
        'Title: Per-pond S, MAE and MAPE, plan 9.1. Each individual horizon and every saved model is retained. Eligible LOPO, descriptive calendar hold-out and descriptive all-events sets are separate. Combined T2 horizons are not duplicated.',
        'Means come from per_pond.csv; sample SDs over five seeds are independently calculated from per_pond_seed.csv. Seed SDs match mean precision. Stopped models have no pond-level metrics and are listed below, not assigned fabricated pond scores.',*stop_notes(stopped),*notes,
    ])

if __name__ == '__main__':
    main()
