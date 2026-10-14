"""Generate Table 10's variants, merged-stream test and pond influence panels."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (horizon_name, model_name, number, result_table, setup, write_table)


def main() -> None:
    result_table('sensitivity','lopo','eligible',('sensitivity',))
    sources,cfg=setup()
    saved=sources.csv('results/stats/merged_units.csv')
    rows=[]
    for r in saved:
        if r['horizon']=='all':
            continue
        rows.append([horizon_name(r['task'],int(r['horizon'])),model_name(r['model']),model_name(r['comparator']),
                     r['n_units'],number(r['S_model'],'s'),number(r['S_comparator'],'s'),r['n_zero_dropped'],
                     f"{float(r['t_plus']):g}",number(r['p_one_sided_lower'],'p'),number(r['p_two_sided'],'p')])
    write_table('sensitivity_merged_units',['Task / horizon','Model','Comparator','Stream units','S model','S comparator','Zeros dropped','T+','Exact one-sided p','Exact two-sided p'],rows,sources,[
        'Title: Descriptive merged-stream sensitivity test, plan 8.3; Table 10 companion. Ponds 1 and 2 are averaged into one unit; other ponds retain one unit each. These exact signed-rank p-values do not replace the confirmatory sequence.',
        'S is averaged over the six resulting units. Combined T2 horizons are not duplicated as an extra row.',
    ])
    sources,cfg=setup()
    saved=sources.csv('results/stats/influence.csv')
    scores=sources.csv('results/metrics/scores.csv')
    nonpositive=[f"{horizon_name(r['task'],int(r['horizon']))}, {model_name(r['model'])}: {int(float(r['n_nonpositive']))} non-positive predictions before removing a pond."
                 for r in scores if r['scheme']=='lopo' and r['sample_set']=='eligible' and r['horizon']!='all' and float(r['n_nonpositive'])>0]
    rows=[[horizon_name(r['task'],int(r['horizon'])),model_name(r['model']),model_name(r['comparator']),r['pond_removed'],r['n_ponds_left'],
           number(r['S_model'],'s'),number(r['S_comparator'],'s'),number(r['diff'],'s')]
          for r in saved if r['horizon']!='all']
    write_table('sensitivity_influence',['Task / horizon','Model','Comparator','Pond removed','Ponds left','S model','S comparator','S difference'],rows,sources,[
        'Title: Descriptive pond influence, plan 8.3; Table 10 companion. Remove each pond from saved per-pond errors without refitting. Difference is model S minus comparator S on the remaining ponds.',
        'Every saved comparison at each individual horizon is retained. Combined T2 horizons are not duplicated. ∞ indicates retained non-positive predictions.',*nonpositive,
    ])

if __name__ == '__main__':
    main()
