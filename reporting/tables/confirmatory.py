"""Generate the fixed-sequence test and the separately labelled permutation panel."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import (MISSING, horizon_name, model_name, number, selections, setup, stops, stop_notes, unique, write_table)


def main() -> None:
    sources, cfg = setup()
    saved = sources.csv('results/stats/confirmatory.csv')
    expected = [(t, str(h)) for t,h in cfg['evaluate']['confirmatory_sequence']]
    if [(r['task'],r['horizon']) for r in saved] != expected:
        raise ValueError('fixed sequence differs from the configuration')
    rows = [[r['step'], horizon_name(r['task'], int(r['horizon'])), model_name(r['model']), model_name(r['comparator']),
             r['n_ponds'], r['n_zero_dropped'], f"{float(r['t_plus']):g}", number(r['p_one_sided_lower'],'p'),
             number(r['p_two_sided'],'p'), r['status']] for r in saved]
    write_table('confirmatory', ['Step','Task / horizon','Model','Comparator','Ponds','Zeros dropped','T+','Exact one-sided p','Exact two-sided p','Decision at alpha 0.05'], rows, sources, [
        'Title: Fixed-sequence confirmatory tests, plan 9.5. One-sided alternative: model S is lower than comparator S. Exact midrank signed-rank distribution, zero differences dropped.',
        'The T2 step is tested only if T1 rejected; its saved exact p-values are descriptive when the gate is closed. Decisions use unrounded p-values at alpha 0.05.',
    ])
    sources, cfg = setup()
    saved = sources.csv('results/stats/permutation.csv')
    primary = selections(sources)
    stopped = stops(sources)
    rows = []
    for task in ('T1','T2'):
        for model in [*cfg['permutation']['tabular_models'][task], primary[task]['primary_deep']]:
            h = 0 if task=='T1' else 1
            prefix = [horizon_name(task,h), model_name(model)]
            if (task,model) in stopped:
                rows.append(prefix + [MISSING]*5 + ['No test: stopped (plan 9.0)'])
            else:
                r = unique(saved,task=task,model=model)
                rows.append(prefix+[r['seed'],number(r['S_obs'],'s'),r['R'],r['n_perm_le_obs'],number(r['p'],'p'),MISSING])
    write_table('permutation', ['Task / horizon','Model','Seed','Observed S','Permutations','Permuted S ≤ observed','Permutation p','Status'], rows, sources, [
        'Title: Within-date water-quality permutation tests, plan 8.2; companion to Table 5. T2 uses the 14-day horizon; deep models use seed 0 for observed and permuted fits.',
        'Permutation tests of plan 8.2, outside the confirmatory sequence of plan 9.5 (no multiplicity adjustment); p = (1 + number of permuted S at or below the observed S) / (R + 1).',
        'Tuned hyperparameters are fixed. T2 Ridge (past weights + WQ) has no test because its frozen run stopped under plan 9.0. Post hoc results are excluded.',
        *stop_notes(stopped),
    ])

if __name__ == '__main__':
    main()
