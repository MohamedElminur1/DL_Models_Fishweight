"""Independent Part B artifact checks. Never import a table generator or its formatters."""

from collections import defaultdict
from decimal import Decimal
from datetime import date, timedelta
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import yaml

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reporting'))
from names import GROUP_NAMES, MODEL_NAMES, horizon_name, SCHEME_NAMES, SAMPLE_SET_NAMES, TASK_NAMES

NAMES = ('main_results','main_results_full','confirmatory','permutation','inner_selection','chosen_hyperparameters','calendar_holdout','all_events',
         'sensitivity','sensitivity_merged_units','sensitivity_influence','per_pond','runtimes','posthoc_ridge','posthoc_ridge_alphas','posthoc_ridge_agreement','sensor_qc')
ORDER = ('time_only','water_quality','reference','deep','static_only','sensitivity')
STOP='stopped (plan 9.0)'
DASH='–'
COMMON={'configs/phase4.yaml','reporting/names.py'}
POST='results/posthoc/ridge_candidate_loses'


def read(path: Path) -> list[dict[str,str]]:
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle))


def get(rows: list[dict[str,str]], **criteria) -> list[dict[str,str]]:
    return [r for r in rows if all(r[k]==str(v) for k,v in criteria.items())]


def one(rows: list[dict[str,str]], **criteria) -> dict[str,str]:
    values=get(rows,**criteria)
    if len(values)!=1:
        raise ValueError(f'expected unique row {criteria}, got {len(values)}')
    return values[0]


def fmt(value, kind: str, places: int | None = None) -> str:
    """Recompute manuscript precision independently, including inapplicable values."""
    if value in (None,''):
        return DASH
    n=float(value)
    if math.isinf(n) and n>0 and kind=='s':
        return '∞'
    if not math.isfinite(n):
        raise ValueError('nonfinite table value')
    if kind=='p':
        if not 0<=n<=1:
            raise ValueError('invalid p')
        return '< 0.001' if n<0.001 else format(n,'.3f')
    if kind in ('grams','percent','r2') and abs(n)>=10000:
        rounded=Decimal(format(n,'.2g'))
        exponent=rounded.copy_abs().adjusted()
        coefficient=format(rounded.scaleb(-exponent),'.1f')
        return f'{coefficient} × 10^{{{exponent}}}'
    decimals=places if places is not None else 3 if kind in ('s','r2') else 1 if kind=='percent' or abs(n)<100 else 0
    return format(n,f'.{decimals}f')


def seed_metric(values: list[float], kind: str) -> str:
    mean=statistics.mean(values)
    text=fmt(mean,kind)
    if len(values)>1:
        text+=' ± '+fmt(statistics.stdev(values),kind,len(text.split(' × ')[0].partition('.')[2]))
    return text


def expect(actual: str, expected: str, label: str) -> None:
    if actual!=expected:
        raise ValueError(f'{label}: got {actual!r}, expected {expected!r}')


def cells(row: dict[str,str], expected: dict[str,str], label: str) -> None:
    for key,value in expected.items():
        expect(row[key],value,f'{label} {key}')


def sidecar(directory: Path,name: str,files: set[str],required: tuple[str,...]=()) -> None:
    content=(directory/f'{name}.source.txt').read_text(encoding='utf-8')
    listing,footer=content.split('\nFooter / definitions / limitations:\n',1)
    paths=listing.splitlines()[1:]
    if set(paths)!=files or len(paths)!=len(files) or not footer.strip():
        raise ValueError(f'{name}: incorrect provenance')
    for path in paths:
        if path.startswith('verification/') or not (ROOT/path).is_file():
            raise ValueError(f'{name}: forbidden or absent source {path}')
    if any(text not in footer for text in required):
        raise ValueError(f'{name}: missing required footer')
    if 'absent in results/metrics/completeness.csv' in footer:
        raise ValueError(f'{name}: incorrect completeness wording')
    if 'results/metrics/completeness.csv' in footer and 'listed without predictions in results/metrics/completeness.csv' not in footer:
        raise ValueError(f'{name}: missing completeness wording')
    if ' × 10^{' in (directory/f'{name}.csv').read_text(encoding='utf-8'):
        if 'two significant digits in builder markup, for example -4.3 × 10^{5}.' not in footer:
            raise ValueError(f'{name}: missing large-value precision note or ASCII minus example')


def check_stop_notes(directory: Path, name: str, rows: list[dict[str,str]], stopped: set[tuple[str,str]]) -> None:
    """Require a stop note exactly when the corresponding stopped run has a row."""
    footer=(directory/f'{name}.source.txt').read_text(encoding='utf-8').split('\nFooter / definitions / limitations:\n',1)[1]
    expected={f'{TASK_NAMES[task]}, {MODEL_NAMES[model]}' for task,model in stopped
              if any(row.get('Model')==MODEL_NAMES[model]
                     and row.get('Task',row.get('Task / horizon','')).startswith(task)
                     and row.get('Evaluation',SCHEME_NAMES['lopo'])==SCHEME_NAMES['lopo']
                     and row.get('Sample set',SAMPLE_SET_NAMES['eligible'])==SAMPLE_SET_NAMES['eligible']
                     for row in rows)}
    notes=[line.split(': stopped (plan 9.0)',1)[0] for line in footer.splitlines()
           if ': stopped (plan 9.0)' in line]
    if set(notes)!=expected or len(notes)!=len(expected):
        raise ValueError(f'{name}: stop notes do not match represented stopped runs')


def model_rows(cfg: dict,task: str,groups: tuple[str, ...]=ORDER) -> list[tuple[str, str]]:
    return [(g,m) for g in groups for m in cfg['models'][task][g]]


def check_sensor_qc(directory: Path, rows: list[dict[str,str]], cfg: dict) -> None:
    """Compare every pond, sensor, labelled unit and count with the saved QC inputs."""
    headers=['Pond','Sensor','Unit as labelled','Readings after duplicate removal','Missing raw readings',
             'Non-finite readings','Out-of-range readings','Flat-line readings','Valid readings',
             'Duplicate rows dropped','Sensor-days','Observed sensor-days','Interpolated sensor-days','Missing sensor-days']
    if list(rows[0])!=headers:
        raise ValueError('sensor_qc: incorrect columns')
    qc=read(ROOT/'results/qc/qc_counts.csv')
    daily=read(ROOT/'results/qc/interpolation_counts.csv')
    schema=read(ROOT/'results/audit/schema.csv')
    identities=[(str(p),s) for p in cfg['design']['cohort'] for s in cfg['design']['sensors']]
    if len(qc)!=len(identities) or len(daily)!=len(identities):
        raise ValueError('sensor_qc: incorrect source pond/sensor row count')
    if len(rows)!=len(identities):
        raise ValueError('sensor_qc: incorrect pond/sensor row count')
    for out,(pond,sensor) in zip(rows,identities):
        counts=one(qc,pond=pond,sensor=sensor)
        days=one(daily,pond=pond,sensor=sensor)
        label=one(schema,pond=pond,canonical_name='ammonia' if sensor=='ammonia_log10' else sensor)
        sensor_name=label['original_name'].partition('(')[0].strip()
        sensor_name='pH' if sensor_name.upper()=='PH' else sensor_name.capitalize()
        expected=[pond,sensor_name,label['unit_as_labelled'] or DASH]
        expected += [counts[k] for k in ('n_values','n_missing_raw','n_nonfinite','n_out_of_range','n_flatline','n_valid','n_duplicate_rows_dropped')]
        expected += [days[k] for k in ('n_days','n_observed','n_interpolated','n_missing')]
        cells(out,dict(zip(headers,expected)),f'sensor_qc pond {pond} {sensor}')
        values=[int(v) for v in expected[3:]]
        if any(v<0 for v in values) or values[0]!=sum(values[1:6]) or values[7]!=sum(values[8:]):
            raise ValueError('sensor_qc: source counts do not balance')
    data_cfg=yaml.safe_load((ROOT/'configs/phase4_data.yaml').read_text(encoding='utf-8'))
    end=date.fromisoformat(data_cfg['cohort']['day0'])+timedelta(days=data_cfg['cohort']['grid']['period_days']*data_cfg['cohort']['grid']['k_last'])
    sidecar(directory,'sensor_qc',{'configs/phase4_data.yaml','ANALYSIS_PLAN.md','results/audit/schema.csv',
            'results/qc/qc_counts.csv','results/qc/interpolation_counts.csv'},(
            'Supplementary sensor quality control',
            'Values count timestamped raw readings after exact duplicate rows are dropped',
            'Missing raw readings are empty or NaN', 'Non-finite readings are positive or negative infinity',
            'non-finite, out of range, then flat-line', 'Valid readings pass all rules',
            'duplicate count is repeated for each sensor of a pond and must not be summed across sensors',
            'temperature 10 to 40;', 'turbidity 0 to 100;', 'pH 0 to 14;',
            'dissolved oxygen > 0 and ≤ 20;', 'ammonia > 0 and ≤ 500;', 'nitrate > 0.',
            'identical consecutive finite raw values spanning at least 24 h',
            'missing or non-finite value or a logging gap of more than 1 h ends a run',
            'valid readings in at least 12 distinct clock hours',
            f"full padded daily grid, {data_cfg['daily']['first_date']} to {end.isoformat()}, inclusive",
            'linearly interpolated only inside runs of at most 2 missing days with observed days on both sides',
            'Missing sensor-days have neither an observed nor an interpolated value',
            'Every interpolated sensor-day is counted here',
            f"Total interpolated sensor-days: {sum(int(r['n_interpolated']) for r in daily)}.",
            'Units are reproduced as labelled in the source, without conversion',
            'ANALYSIS_PLAN.md sections 2.5 to 2.7',))


def check(directory: Path,cfg: dict) -> None:
    """Check all data cells in most panels, with independent seed and pond aggregation."""
    tables={}
    for name in ('calendar_holdout','all_events','sensitivity'):
        if any((directory/f'{name}_full{suffix}').exists() for suffix in ('.csv','.source.txt')):
            raise ValueError(f'{name}: full version is outside revision 3')
    for name in NAMES:
        path=directory/f'{name}.csv'
        with path.open(encoding='utf-8',newline='') as handle:
            reader=csv.DictReader(handle)
            header=reader.fieldnames
            rows=list(reader)
        if not header or len(set(header))!=len(header) or not rows or any(None in r or None in r.values() for r in rows):
            raise ValueError(f'{name}: malformed table')
        for row in [dict(zip(header,header)),*rows]:
            if any('\n' in c or '\r' in c for c in row.values()):
                raise ValueError(f'{name}: multiline cell')
            if any(c=='NA' for c in row.values()):
                raise ValueError(f'{name}: forbidden NA cell')
            count=sum(c.count(STOP) for c in row.values())
            if count and (count!=1 or STOP not in row.values()):
                raise ValueError(f'{name}: stopped label must occur once in a statistic cell')
        tables[name]=rows
    for path in directory.glob('*.csv'):
        if '−' in path.read_text(encoding='utf-8'):
            raise ValueError(f'{path.stem}: negative numbers require ASCII hyphen-minus')
    saved=lambda p:read(ROOT/p)
    primary={t:json.loads((ROOT/f'results/selection/primary_{t}.json').read_text()) for t in ('T1','T2')}
    selection_files={f'results/selection/primary_{t}.json' for t in primary}
    complete=saved('results/metrics/completeness.csv')
    log=(ROOT/'results/logs/autopilot.log').read_text()
    stopped={(r['task'],r['model']) for r in complete if r['scheme']=='lopo' and r['has_predictions']=='False'}
    for task,model in stopped:
        if not any('stopped under plan 9.0 (non-finite prediction)' in line and f'--task {task} --scheme lopo --model {model} ' in line for line in log.splitlines()):
            raise ValueError('missing logged stop')
    stop_files={'results/metrics/completeness.csv','results/logs/autopilot.log'}
    for name,rows in tables.items():
        check_stop_notes(directory,name,rows,stopped if name!='calendar_holdout' else set())
    scores=saved('results/metrics/scores.csv')
    pooled=saved('results/metrics/pooled.csv')
    perpond=saved('results/metrics/per_pond.csv')
    perseed=saved('results/metrics/per_pond_seed.csv')
    poolseed=saved('results/metrics/pooled_by_seed.csv')
    bootstrap=saved('results/stats/bootstrap_S.csv')
    for name,scheme,sample_set,groups in [('main_results','lopo','eligible',ORDER[:-1]),('main_results_full','lopo','eligible',ORDER),('calendar_holdout','chrono','eligible',ORDER[:-1]),
                                          ('all_events','lopo','all',('all_sample_set',)),('sensitivity','lopo','eligible',('sensitivity',))]:
        compact=name!='main_results_full'
        full_header=['Task / horizon','Group','Model','Primary baseline','Ponds','Samples','Seeds','S','S bootstrap 95 % interval',
                     'MAE (g)','RMSE (g)','MAPE (%)','R^{2}','Ponds with lower S','SD across ponds of S','SD across ponds of MAE (g)']
        expected_header=[c for c in full_header if c not in ('Primary baseline','Ponds','Samples','Seeds','SD across ponds of S','SD across ponds of MAE (g)')]
        expected=[]
        for task in ('T1','T2'):
            for h in ([0] if task=='T1' else [1,2]):
                for group,model in model_rows(cfg,task,groups):
                    expected.append((task,h,group,model))
        panels=list(dict.fromkeys((task,h) for task,h,_,_ in expected))
        denominators={len(get(perpond,task=t,horizon=h,scheme=scheme,sample_set=sample_set,
                              model=primary[t]['primary_baseline'])) for t,h in panels}
        denominator=str(next(iter(denominators))) if len(denominators)==1 else 'N'
        comparison=f'Ponds below the primary baseline (of {denominator})'
        expected_header[-1]=comparison
        if list(tables[name][0])!=(expected_header if compact else full_header):
            raise ValueError(f'{name}: incorrect result columns')
        if len(tables[name])!=len(expected):
            raise ValueError(f'{name}: incomplete model rows')
        for out,(task,h,group,model) in zip(tables[name],expected):
            baseline=primary[task]['primary_baseline']
            identity={'Task / horizon':horizon_name(task,h),'Group':GROUP_NAMES[group],'Model':MODEL_NAMES[model]}
            if not compact:
                identity['Primary baseline']=MODEL_NAMES[baseline]
            cells(out,identity,name)
            if scheme=='lopo' and (task,model) in stopped:
                for key in list(out)[3 if compact else 4:]:
                    expect(out[key],STOP if key=='S' else DASH,f'{name} stopped {key}')
                continue
            criteria=dict(task=task,scheme=scheme,sample_set=sample_set,horizon=h,model=model)
            pond_rows=get(perpond,**criteria)
            seed_rows=get(perseed,**criteria)
            metrics=get(poolseed,**criteria)
            byseed=defaultdict(list)
            for r in seed_rows:
                byseed[r['seed']].append(float(r['S']))
            seeds=cfg['seeds']['final'] if model in cfg['deep']['arch_space'] else [0]
            if set(map(int,byseed))!=set(seeds):
                raise ValueError(f'{name}: incomplete seeds')
            seed_s=[statistics.mean(byseed[str(seed)]) for seed in seeds]
            base={r['pond']:float(r['S']) for r in get(perpond,**(criteria|{'model':baseline}))}
            if {r['pond'] for r in pond_rows}!=set(base):
                raise ValueError(f'{name}: model and baseline have different scored ponds')
            count=sum(float(r['S'])<base[r['pond']] for r in pond_rows)
            interval=DASH
            if scheme=='lopo' and sample_set=='eligible':
                b=one(bootstrap,task=task,horizon=h,model=model)
                interval=f"{fmt(b['ci_low'],'s')} to {fmt(b['ci_high'],'s')}"
            wanted={'S':seed_metric(seed_s,'s'),'S bootstrap 95 % interval':interval,
                    comparison if compact else 'Ponds with lower S':str(count)}
            if not compact:
                wanted.update({'Ponds':str(len(pond_rows)),'Samples':str(sum(int(r['n']) for r in pond_rows)),'Seeds':str(len(seeds)),
                               'SD across ponds of S':fmt(statistics.stdev(float(r['S']) for r in pond_rows),'s'),
                               'SD across ponds of MAE (g)':fmt(statistics.stdev(float(r['mae_g']) for r in pond_rows),'grams')})
            cells(out,wanted,name)
            for key,column,kind in [('mae_g','MAE (g)','grams'),('rmse_g','RMSE (g)','grams'),('mape_pct','MAPE (%)','percent'),('r2','R^{2}','r2')]:
                expect(out[column],seed_metric([float(one(metrics,seed=seed)[key]) for seed in seeds],kind),f'{name} {model} {column}')
        files=COMMON|selection_files|stop_files|{'results/metrics/scores.csv','results/metrics/pooled.csv','results/metrics/per_pond.csv'}
        if scheme=='lopo' and sample_set=='eligible':
            files.add('results/stats/bootstrap_S.csv')
        required=('Supplementary',) if name=='main_results_full' else ('Descriptive',) if name!='main_results' else ('stopped (plan 9.0)', 'Deep models and the static MLP show mean ± SD over five seeds.')
        if compact:
            required+=('Deep models and the static MLP show mean ± SD over five seeds.',)
            for task,h in panels:
                baseline=primary[task]['primary_baseline']
                pp=get(perpond,task=task,horizon=h,scheme=scheme,sample_set=sample_set,model=baseline)
                required+=(f'{horizon_name(task,h)}: primary baseline {MODEL_NAMES[baseline]}; {len(pp)} ponds; {sum(int(r["n"]) for r in pp)} samples.',)
                if denominator=='N':
                    required+=(f'{horizon_name(task,h)}: N = {len(pp)} scored ponds.',)
        sidecar(directory,name,files,required)
    check_sensor_qc(directory,tables['sensor_qc'],cfg)
    confirm=saved('results/stats/confirmatory.csv')
    if [(r['task'],int(r['horizon'])) for r in confirm]!=[tuple(pair) for pair in cfg['evaluate']['confirmatory_sequence']] or len(tables['confirmatory'])!=len(confirm):
        raise ValueError('confirmatory: wrong sequence')
    for out,r in zip(tables['confirmatory'],confirm):
        cells(out,{'Step':r['step'],'Task / horizon':horizon_name(r['task'],int(r['horizon'])),'Model':MODEL_NAMES[r['model']],
                   'Comparator':MODEL_NAMES[r['comparator']],'Ponds':r['n_ponds'],'Zeros dropped':r['n_zero_dropped'],
                   'T+':format(float(r['t_plus']),'g'),'Exact one-sided p':fmt(r['p_one_sided_lower'],'p'),
                   'Exact two-sided p':fmt(r['p_two_sided'],'p'),'Decision at alpha 0.05':r['status']},'confirmatory')
    sidecar(directory,'confirmatory',COMMON|{'results/stats/confirmatory.csv'},('plan 9.5','only if T1 rejected'))
    permutation=saved('results/stats/permutation.csv')
    identities=[(t,m) for t in primary for m in [*cfg['permutation']['tabular_models'][t],primary[t]['primary_deep']]]
    if len(tables['permutation'])!=len(identities):
        raise ValueError('permutation: incomplete models')
    for out,(task,model) in zip(tables['permutation'],identities):
        cells(out,{'Task / horizon':horizon_name(task,0 if task=='T1' else 1),'Model':MODEL_NAMES[model]},'permutation')
        if (task,model) in stopped:
            for k in list(out)[2:]:
                expect(out[k],STOP if k=='Observed S' else DASH,'permutation stop')
        else:
            r=one(permutation,task=task,model=model)
            cells(out,{'Seed':r['seed'],'Observed S':fmt(r['S_obs'],'s'),'Permutations':r['R'],'Permuted S ≤ observed':r['n_perm_le_obs'],
                       'Permutation p':fmt((1+int(r['n_perm_le_obs']))/(1+int(r['R'])),'p'),'Status':DASH},'permutation')
    sidecar(directory,'permutation',COMMON|selection_files|stop_files|{'results/stats/permutation.csv'},('no test','Post hoc results are excluded',
            'Permutation tests of plan 8.2, outside the confirmatory sequence of plan 9.5 (no multiplicity adjustment); p = (1 + number of permuted S at or below the observed S) / (R + 1).'))
    innerfiles=COMMON|selection_files|stop_files
    identities=[(t,g,m) for t in primary for g,m in model_rows(cfg,t)]
    if len(tables['inner_selection'])!=len(identities):
        raise ValueError('inner_selection: incomplete models')
    for out,(task,group,model) in zip(tables['inner_selection'],identities):
        cells(out,{'Task':TASK_NAMES[task],'Group':GROUP_NAMES[group],'Model':MODEL_NAMES[model]},'inner_selection')
        if (task,model) in stopped:
            cells(out,{'Outer folds':DASH,'Mean inner-validation S':STOP,'Selection':DASH},'inner_selection stop')
            continue
        path=f'results/tuning/{task}/lopo/{model}/inner_scores.csv'
        innerfiles.add(path)
        folds=defaultdict(list)
        for r in saved(path):
            folds[r['outer_fold']].append(float(r['S']))
        means={int(f):statistics.mean(v) for f,v in folds.items()}
        if set(means)!=set(primary[task]['outer_folds']):
            raise ValueError('inner_selection: incomplete folds')
        mean=statistics.mean(means.values())
        if group in ('time_only','deep'):
            if not math.isclose(mean,primary[task]['mean_inner_S'][model],rel_tol=1e-12):
                raise ValueError('inner selection and committed selection differ')
            pool=cfg['models'][task][group]
            winner=min(pool,key=lambda m:primary[task]['mean_inner_S'][m])
            chosen=primary[task]['primary_baseline' if group=='time_only' else 'primary_deep']
            if winner!=chosen:
                raise ValueError('wrong primary selection')
            label=('Primary baseline' if group=='time_only' else 'Primary deep model') if model==chosen else 'Not selected'
        else:
            label='Outside primary selection pools'
        cells(out,{'Outer folds':str(len(means)),'Mean inner-validation S':fmt(mean,'s'),'Selection':label},'inner_selection')
    sidecar(directory,'inner_selection',innerfiles,('plan 9.4',))
    chosenfiles=COMMON|stop_files
    expected=[]
    for task in primary:
        entries=[(g,m,'eligible') for g,m in model_rows(cfg,task)]+[('all_sample_set',m,'all') for m in cfg['models'][task]['all_sample_set']]
        for group,model,sample_set in entries:
            prefix={'Task':TASK_NAMES[task],'Sample set':SAMPLE_SET_NAMES[sample_set],'Group':GROUP_NAMES[group],'Model':MODEL_NAMES[model]}
            if (task,model) in stopped:
                expected.append(prefix|{'Outer fold':STOP,'Held-out pond':DASH,'Chosen trial':DASH,'Final-fit epochs':DASH,'Parameter':DASH,'Chosen value':DASH})
                continue
            path=f"results/tuning/{task}/lopo/{model}{'__all' if sample_set=='all' else ''}/chosen.csv"
            chosenfiles.add(path)
            for r in sorted(saved(path),key=lambda r:int(r['outer_fold'])):
                base=prefix|{'Outer fold':r['outer_fold'],'Held-out pond':r['test_ponds'],'Chosen trial':r['best_trial'] or DASH,'Final-fit epochs':r['epochs'] or DASH}
                parameters=[k for k in r if k.startswith('hp_') and r[k]]
                for p in parameters or [None]:
                    key=p[3:].replace('_','\\_') if p else DASH
                    expected.append(base|{'Parameter':key,'Chosen value':r[p] if p else DASH})
    if len(expected)!=len(tables['chosen_hyperparameters']):
        raise ValueError('chosen_hyperparameters: incomplete parameters')
    for out,want in zip(tables['chosen_hyperparameters'],expected):
        cells(out,want,'chosen_hyperparameters')
    sidecar(directory,'chosen_hyperparameters',chosenfiles,('Stopped runs',))
    merged=[r for r in saved('results/stats/merged_units.csv') if r['horizon']!='all']
    if len(tables['sensitivity_merged_units'])!=len(merged):
        raise ValueError('merged units: missing rows')
    for out,r in zip(tables['sensitivity_merged_units'],merged):
        cells(out,{'Task / horizon':horizon_name(r['task'],int(r['horizon'])),'Model':MODEL_NAMES[r['model']],'Comparator':MODEL_NAMES[r['comparator']],
                   'Stream units':r['n_units'],'S model':fmt(r['S_model'],'s'),'S comparator':fmt(r['S_comparator'],'s'),'Zeros dropped':r['n_zero_dropped'],
                   'T+':format(float(r['t_plus']),'g'),'Exact one-sided p':fmt(r['p_one_sided_lower'],'p'),'Exact two-sided p':fmt(r['p_two_sided'],'p')},'merged_units')
    sidecar(directory,'sensitivity_merged_units',COMMON|{'results/stats/merged_units.csv'},('Descriptive','Ponds 1 and 2'))
    influence=[r for r in saved('results/stats/influence.csv') if r['horizon']!='all']
    if len(tables['sensitivity_influence'])!=len(influence):
        raise ValueError('influence: missing rows')
    for out,r in zip(tables['sensitivity_influence'],influence):
        criteria={k:r[k] for k in ('task','horizon')}|{'scheme':'lopo','sample_set':'eligible'}
        a=[float(p['S']) for p in get(perpond,**criteria,model=r['model']) if p['pond']!=r['pond_removed']]
        b=[float(p['S']) for p in get(perpond,**criteria,model=r['comparator']) if p['pond']!=r['pond_removed']]
        cells(out,{'Task / horizon':horizon_name(r['task'],int(r['horizon'])),'Model':MODEL_NAMES[r['model']],'Comparator':MODEL_NAMES[r['comparator']],
                   'Pond removed':r['pond_removed'],'Ponds left':str(len(a)),'S model':fmt(statistics.mean(a),'s'),
                   'S comparator':fmt(statistics.mean(b),'s'),'S difference':fmt(statistics.mean(a)-statistics.mean(b),'s')},'influence')
    sidecar(directory,'sensitivity_influence',COMMON|{'results/stats/influence.csv','results/metrics/scores.csv'},('Descriptive','without refitting'))
    rows=[r for r in perpond if r['horizon']!='all']
    if len(tables['per_pond'])!=len(rows):
        raise ValueError('per_pond: missing rows')
    for out,r in zip(tables['per_pond'],rows):
        group='all_sample_set' if r['sample_set']=='all' else next(g for g,m in model_rows(cfg,r['task']) if m==r['model'])
        criteria={k:r[k] for k in ('task','scheme','sample_set','horizon','model','pond')}
        values=get(perseed,**criteria)
        cells(out,{'Task / horizon':horizon_name(r['task'],int(r['horizon'])),'Evaluation':SCHEME_NAMES[r['scheme']],
                   'Sample set':SAMPLE_SET_NAMES[r['sample_set']],'Group':GROUP_NAMES[group],'Model':MODEL_NAMES[r['model']],
                   'Pond':r['pond'],'Samples':r['n'],'Seeds':r['n_seeds'],
                   **{column:seed_metric([float(v[key]) for v in values],kind) for key,column,kind in [('S','S','s'),('mae_g','MAE (g)','grams'),('mape_pct','MAPE (%)','percent')]}},'per_pond')
    sidecar(directory,'per_pond',COMMON|stop_files|{'results/metrics/per_pond.csv','results/metrics/per_pond_seed.csv'},('sample SDs','Stopped models have no pond-level metrics'))
    runtime=saved('results/logs/runtime.csv')
    groups=defaultdict(list)
    for r in runtime:
        if r['scheme']!='permutation':
            groups[(r['task'],r['scheme'],r['model'])].append(r)
    if len(tables['runtimes'])!=len(groups):
        raise ValueError('runtimes: incomplete runs')
    for out,(identity,part) in zip(tables['runtimes'],sorted(groups.items())):
        task,scheme,code=identity
        model=code.removesuffix('__all')
        is_stopped=(task,model) in stopped and scheme=='lopo'
        stages={stage:[float(r['seconds']) for r in part if r['stage']==stage] for stage in ('trial','final')}
        perm=[float(r['seconds']) for r in runtime if r['stage']=='permutation' and r['task']==task and r['model']==code] if scheme=='lopo' and not code.endswith('__all') else []
        cells(out,{'Task':TASK_NAMES[task],'Evaluation':SCHEME_NAMES[scheme],'Sample set':SAMPLE_SET_NAMES['all' if code.endswith('__all') else 'eligible'],'Model':MODEL_NAMES[model],
                   'Tuning time (s)':STOP if is_stopped else format(math.fsum(stages['trial']),'.1f') if stages['trial'] else DASH,
                   'Final-fit time (s)':DASH if is_stopped else format(math.fsum(stages['final']),'.1f') if stages['final'] else DASH,
                   'Permutation time (s)':DASH if is_stopped else format(math.fsum(perm),'.1f') if perm else DASH,
                   'CPU':part[0]['cpu'],'Python':part[0]['python'],'PyTorch':part[0]['torch'],
                   'Status':DASH if is_stopped else 'Complete'},'runtimes')
    sidecar(directory,'runtimes',COMMON|stop_files|{'results/logs/runtime.csv'},('parallel workers can overlap','No GPU, RAM'))
    post=[r for r in saved(f'{POST}/metrics/scores.csv') if r['horizon']!='all']
    postperm=saved(f'{POST}/stats/permutation.csv')
    if len(tables['posthoc_ridge'])!=len(post):
        raise ValueError('posthoc_ridge: missing variants')
    for out,r in zip(tables['posthoc_ridge'],post):
        counterpart='ridge_time' if r['task']=='T1' else 'ridge_pw'
        base=one(scores,task=r['task'],scheme='lopo',sample_set='eligible',horizon=r['horizon'],model=counterpart)
        perm=get(postperm,task=r['task'],model=r['model'])
        p=fmt((int(perm[0]['n_perm_le_obs'])+1)/(int(perm[0]['R'])+1),'p') if perm and r['horizon']=='1' else DASH
        cells(out,{'Task / horizon':horizon_name(r['task'],int(r['horizon'])),'Post hoc model':MODEL_NAMES[r['model']],
                   'Ponds':r['n_ponds'],'Post hoc S':fmt(r['S'],'s'),'Frozen counterpart':MODEL_NAMES[counterpart],
                   'Frozen S':fmt(base['S'],'s'),'Post hoc permutation p':p,
                   'Test status':'Post hoc permutation at 14 days' if p!=DASH else 'No permutation test for this model / horizon'},'posthoc_ridge')
    sidecar(directory,'posthoc_ridge',COMMON|{'DECISIONS.md',f'{POST}/metrics/scores.csv',f'{POST}/stats/permutation.csv','results/metrics/scores.csv'},('Post hoc','no frozen table 4 to 12'))
    identities=list(dict.fromkeys((r['task'],r['model']) for r in post))
    files=COMMON|{f'{POST}/metrics/scores.csv'}
    alphas=[]
    for task,model in identities:
        path=f'{POST}/tuning/{task}/lopo/{model}/chosen.csv'
        files.add(path)
        alphas.extend({'Task':TASK_NAMES[task],'Post hoc model':MODEL_NAMES[model],'Outer fold':r['outer_fold'],'Held-out pond':r['test_ponds'],'Chosen alpha':r['hp_alpha']} for r in saved(path))
    if len(tables['posthoc_ridge_alphas'])!=len(alphas):
        raise ValueError('posthoc_ridge_alphas: missing folds')
    for out,want in zip(tables['posthoc_ridge_alphas'],alphas):
        cells(out,want,'posthoc_ridge_alphas')
    sidecar(directory,'posthoc_ridge_alphas',files,('Post hoc',))
    agreement=saved(f'{POST}/metrics/codex_agreement.csv')
    if len(tables['posthoc_ridge_agreement'])!=len(agreement):
        raise ValueError('posthoc_ridge_agreement: missing folds')
    for out,r in zip(tables['posthoc_ridge_agreement'],agreement):
        cells(out,{'Task':TASK_NAMES['T2'],'Post hoc model':MODEL_NAMES['ridge_pw_wq'],'Outer fold':r['outer_fold'],
                   'Held-out pond':r['held_out_pond'],'Chosen alpha':r['alpha'],'Codex alpha':r['codex_alpha'],
                   'Alphas equal':'Yes' if float(r['alpha'])==float(r['codex_alpha']) else 'No','Predictions':r['n_predictions'],
                   'Max absolute log-prediction difference':r['max_abs_y_pred_log'],'Max absolute weight difference (g)':r['max_abs_w_pred']},'posthoc_ridge_agreement')
    sidecar(directory,'posthoc_ridge_agreement',COMMON|{'DECISIONS.md',f'{POST}/metrics/codex_agreement.csv'},('Post hoc','exact saved precision'))
    for name in NAMES:
        if not name.startswith('posthoc_') and POST in (directory/f'{name}.source.txt').read_text():
            raise ValueError(f'{name}: post hoc contamination')
    print(f'PASS: independent Part B cell recomputation, complete rows, stops, seed precision and provenance for {len(NAMES)} panels')
