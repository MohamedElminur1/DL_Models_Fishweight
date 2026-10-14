"""Generate Table 13's post hoc scores, chosen alphas and available rebuild agreement."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (MISSING, TASK_NAMES, horizon_name, model_name, number, select, setup, unique, write_table)

POSTHOC='results/posthoc/ridge_candidate_loses'


def main() -> None:
    sources,cfg=setup()
    decision=sources.text('DECISIONS.md')
    if 'D-003' not in decision or 'infinite inner score' not in decision:
        raise ValueError('missing post hoc decision')
    saved=sources.csv(f'{POSTHOC}/metrics/scores.csv')
    frozen=sources.csv('results/metrics/scores.csv')
    permutation=sources.csv(f'{POSTHOC}/stats/permutation.csv')
    rows=[]
    for r in saved:
        if r['horizon']=='all':
            continue
        task,model,h=r['task'],r['model'],int(r['horizon'])
        comparator='ridge_time' if task=='T1' else 'ridge_pw'
        base=unique(frozen,task=task,scheme='lopo',sample_set='eligible',horizon=h,model=comparator)
        perm=select(permutation,task=task,model=model)
        applicable=bool(perm) and h==1
        rows.append([horizon_name(task,h),model_name(model),r['n_ponds'],number(r['S'],'s'),model_name(comparator),number(base['S'],'s'),
                     number(perm[0]['p'],'p') if applicable else MISSING,
                     'Post hoc permutation at 14 days' if applicable else 'No permutation test for this model / horizon'])
    notes=['Title: Post hoc ridge variant, DECISIONS.md D-003; Supplementary Table 13. A non-finite inner-validation candidate loses with infinite S; all other frozen settings remain. These results enter no frozen table 4 to 12.',
           'Every post hoc model and individual horizon is retained, compared with the frozen ridge counterpart without water quality, not the selected primary baseline. Only T2 Ridge (past weights + WQ) at 14 days has a saved post hoc permutation test. – means no applicable test.',
           'Chosen alphas and the available agreement with an independent re-implementation are in the post hoc companion panels. The agreement file covers T2 Ridge (past weights + WQ) only; no saved agreement for the other two variants is supplied.']
    write_table('posthoc_ridge',['Task / horizon','Post hoc model','Ponds','Post hoc S','Frozen counterpart','Frozen S','Post hoc permutation p','Test status'],rows,sources,notes)
    sources,cfg=setup()
    rows=[]
    alpha_scores=sources.csv(f'{POSTHOC}/metrics/scores.csv')
    identities=list(dict.fromkeys((r['task'],r['model']) for r in alpha_scores))
    for task,model in identities:
        chosen=sources.csv(f'{POSTHOC}/tuning/{task}/lopo/{model}/chosen.csv')
        rows.extend([TASK_NAMES[task],model_name(model),r['outer_fold'],r['test_ponds'],r['hp_alpha']] for r in chosen)
    write_table('posthoc_ridge_alphas',['Task','Post hoc model','Outer fold','Held-out pond','Chosen alpha'],rows,sources,[
        'Title: Post hoc chosen ridge alphas, DECISIONS.md D-003; Table 13 companion. Every complete post hoc chosen.csv is included, with its exact saved alpha. Nothing enters frozen tables.',
    ])
    sources,cfg=setup()
    agreement_scope=sources.text('DECISIONS.md')
    if 'its T2 `ridge_pw_wq` alphas and' not in agreement_scope:
        raise ValueError('agreement model scope is not supplied by D-003')
    saved=sources.csv(f'{POSTHOC}/metrics/codex_agreement.csv')
    rows=[[TASK_NAMES['T2'],model_name('ridge_pw_wq'),r['outer_fold'],r['held_out_pond'],r['alpha'],r['codex_alpha'],
           'Yes' if r['alpha_equal']=='True' else 'No',r['n_predictions'],r['max_abs_y_pred_log'],r['max_abs_w_pred']] for r in saved]
    write_table('posthoc_ridge_agreement',['Task','Post hoc model','Outer fold','Held-out pond','Chosen alpha','Re-implementation alpha','Alphas equal','Predictions','Max absolute log-prediction difference','Max absolute weight difference (g)'],rows,sources,[
        'Title: Post hoc agreement with an independent re-implementation, DECISIONS.md D-003; Table 13 companion. Only the saved metrics/codex_agreement.csv is read. Agreement is available for T2 Ridge (past weights + WQ), both horizons pooled within each fold; the file has no model column and the scope is documented by D-003.',
        'Difference values preserve the exact saved precision, including differences below 0.000000000002 g, to distinguish numerical round-off from exact zero. The frozen independent verification scores are never read. Nothing enters tables 4 to 12.',
    ])

if __name__ == '__main__':
    main()
