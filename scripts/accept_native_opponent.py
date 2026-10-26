"""Audit two complete native confirmation sets for one pinned opponent/profile.

This creates evidence of version-specific advantage, never global goal completion
or automatic promotion of the learner over its teacher.
"""
import argparse,collections,gzip,hashlib,json,math,random,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--report',action='append',required=True)
p.add_argument('--audit',action='append',required=True);p.add_argument('--source',required=True)
p.add_argument('--opponent');p.add_argument('--commit');p.add_argument('--candidate');p.add_argument('--candidate-net');p.add_argument('--baseline');p.add_argument('--run-prefix',default='')
p.add_argument('--node');p.add_argument('--out',required=True);a=p.parse_args()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
plan=json.loads(Path(a.plan).read_text());out=Path(a.out)
pins=plan['sha256'] if 'sha256' in plan else plan['source_and_artifact_hashes']
opponent=a.opponent or plan.get('opponent');commit=a.commit or plan.get('commit')
candidate=a.candidate or plan.get('candidate_binary');candidate_net=a.candidate_net or plan.get('candidate_net');baseline=a.baseline or plan.get('baseline')
clock=plan['time_control'].split()[0]
if not all([opponent,commit,candidate,candidate_net,baseline]):raise SystemExit('Missing explicit opponent/artifact identities')
if out.exists():raise SystemExit('Refuse to overwrite acceptance evidence')
if len(plan['sets'])!=2 or len(a.report)!=2 or len(a.audit)!=2:raise SystemExit('Require two complete fixed sets')
for f,digest in pins.items():
 if sha(f)!=digest:raise SystemExit('Frozen artifact changed: '+f)
if 'node_sha256' in plan and (not a.node or sha(a.node)!=plan['node_sha256']):raise SystemExit('Runtime hash mismatch')
if subprocess.check_output(['git','-C',a.source,'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',a.source,'diff','HEAD'],text=True).strip():raise SystemExit('Opponent source revision or working tree differs')
sets=[];all_seeds=set()
for subset,reportfile,auditfile in zip(plan['sets'],a.report,a.audit):
 report=json.loads(Path(reportfile).read_text());audit=json.loads(Path(auditfile).read_text())
 if len(report['reports'])!=1:raise SystemExit('Ambiguous result report')
 r=report['reports'][0];s=r['result'];b=r['budget']
 games=subset.get('games',subset.get('games_per_arm'))
 if r['run']!=a.run_prefix+subset['id'] or r['stage']!='confirmation' or b['seed']!=subset['seed'] or b['games']!=games or b['tc']!=clock or b['concurrency']!=plan['concurrency'] or r['candidate_options']:
  raise SystemExit('Set configuration differs from frozen plan')
 if r['source_referee_commit']!='331a6e612a038c4a04cf980f7b9482910fd50078':raise SystemExit('Wrong native referee revision')
 expected_hashes={'candidate':candidate,'candidate_net':candidate_net,'baseline':baseline}
 for key,file in expected_hashes.items():
  if r['hashes'][key]!=pins[file]:raise SystemExit('Recorded played artifact differs')
 raw=r['raw_games'];data=gzip.decompress(Path(raw['path']).read_bytes())
 digest=hashlib.sha256(data).hexdigest()
 if digest!=raw['uncompressed_sha256'] or digest!=audit['raw_games_sha256']:raise SystemExit('Raw/replay artifact mismatch')
 rows=[json.loads(line) for line in data.splitlines()];pairs=collections.defaultdict(list);scores=[]
 for row in rows:
  if {row['red'],row['black']}!={'candidate','baseline'} or row['result'] not in (0,.5,1):raise SystemExit('Invalid players or result')
  score=row['result'] if row['red']=='candidate' else 1-row['result'];scores.append(score);pairs[row['pair']].append(row)
 if len(rows)!=games or len(pairs)!=len(rows)//2 or len(pairs)<100:raise SystemExit('Incomplete confirmation set')
 values=[];seeds=set()
 for group in pairs.values():
  if len(group)!=2 or {g['red'] for g in group}!={'candidate','baseline'} or len({g['seed'] for g in group})!=1:raise SystemExit('Missing or duplicated color swap')
  seeds.add(group[0]['seed'])
  values.append(statistics.mean(g['result'] if g['red']=='candidate' else 1-g['result'] for g in group))
 if len(seeds)!=len(pairs) or seeds&all_seeds:raise SystemExit('Duplicate held-out deal seeds')
 all_seeds|=seeds
 wdl=[scores.count(x) for x in [1,.5,0]]
 if wdl!=[s[k] for k in ['wins','draws','losses']]:raise SystemExit('WDL mismatch')
 rng=random.Random(20261026)
 boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000));ci=[boot[249],boot[9749]]
 radius=math.sqrt(math.log(40)/(2*len(values)));lower=max(0,statistics.mean(values)-radius)
 if any(abs(x-y)>1e-12 for x,y in zip(ci,s['paired_bootstrap_95'])) or abs(lower-s['paired_hoeffding_95'][0])>1e-12:raise SystemExit('Recomputed intervals disagree')
 if min(ci)<=.5 or lower<=.5 or not s['valid_technical_run'] or s['void_attempts'] or s['technical_results'] or not s['single_match_superiority_gate']:raise SystemExit('Confirmation criterion failed')
 if audit['games']!=len(rows) or audit['complete_pairs']!=len(pairs) or audit['wins_draws_losses']!=wdl:raise SystemExit('Replay counts disagree')
 if audit['replay_referee_library_sha256']!='4c3ff591f0ecfe4cf959ad5999fea594ff90f8cba0dfaeca308d0576881a6c2e':raise SystemExit('Unexpected replay rule binary')
 for flag in ['seeded_identities_match','all_recorded_moves_legal','nontechnical_terminal_results_match']:
  if audit[flag] is not True:raise SystemExit('Replay did not certify '+flag)
 moves=sum(len(row['moves']) for row in rows)
 if audit['moves_replayed']!=moves or audit['actual_clock_moves_checked']!=moves or audit['technical_games'] or audit['clock_ambiguous_moves_from_rounding'] or audit['minimum_clock_lower_margin_ms']<0:raise SystemExit('Incomplete or ambiguous move/clock replay')
 sets.append({'id':subset['id'],'games':len(rows),'pairs':len(pairs),'wins_draws_losses':wdl,'score':statistics.mean(scores),'paired_bootstrap_95':ci,'paired_hoeffding_lower_95':lower,'moves_replayed':moves,'result_path':reportfile,'result_sha256':sha(reportfile),'audit_path':auditfile,'audit_sha256':sha(auditfile),'raw_sha256':digest})
result={'schema':1,'opponent':opponent,'commit':commit,'plan_path':a.plan,'plan_sha256':sha(a.plan),'clock':clock,
 'candidate_binary':candidate,'candidate_net':candidate_net,'baseline':baseline,
 'accepted':True,'scope':'Statistical advantage over this pinned opponent and adapter under the documented external referee and clock. Both fixed sets pass separately, without technical failures, voids or ambiguous clock replays. This is not a universal rules/time-control claim, a general learned improvement over the teacher, or completion of the all-engine goal.',
 'all_frozen_hashes_match_after_completion':True,'unchanged_opponent_source':True,'sets':sets,'total_games':sum(x['games'] for x in sets),'distinct_deal_seeds':len(all_seeds),
 'acceptance_script_sha256':sha(__file__)}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
