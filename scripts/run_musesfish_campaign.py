"""Fixed clock-profile coverage campaign; technical failures stop confirmation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    digest=sha(a.plan);plan=json.loads(Path(a.plan).read_text());out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite or restart campaign')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
        if subprocess.check_output(['git','-C',str(ROOT/'external/musesfish'),'rev-parse','HEAD'],text=True).strip()!=plan['opponent_commit']:
            raise ValueError('Opponent revision changed')
        if subprocess.check_output(['git','-C',str(ROOT/'external/musesfish'),'diff','HEAD'],text=True).strip():
            raise ValueError('Opponent tracked source modified')
    verify();out.mkdir(parents=True)
    state={'schema':1,'pid':os.getpid(),'status':'running','plan_sha256':digest,'completed':[]}
    def save():
        tmp=out/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
    env=dict(os.environ,JIEQI_PYTHON=sys.executable)
    def run(cmd,log):
        with log.open('x') as handle:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            if child.wait():raise RuntimeError('Campaign child failed; see '+log.name)
        state['child_pid']=None
    save()
    try:
        for match in plan['matches']:
            verify();state['stage']=match['id'];save();folder=out/match['id']
            run([sys.executable,'scripts/native_match.py','--candidate='+plan['candidate'],
                '--candidate-net='+plan['candidate_net'],'--net='+plan['candidate_net'],
                '--baseline=adapters/musesfish_clock.sh','--baseline-strict','--baseline-classical',
                '--tc='+plan['clock'],'--games='+str(match['games']),'--seed='+str(match['seed']),
                '--concurrency=2','--stage='+match['stage'],'--out='+str(folder)],out/(match['id']+'.log'))
            verify()
            report='results/20261026-musesfish-clock-'+match['id']
            run([sys.executable,'scripts/archive_native_results.py','--out',report+'.json',
                 '--raw-dir','results/raw',str(folder)],out/(match['id']+'-archive.log'))
            run([sys.executable,'scripts/audit_native_games.py','--run',str(folder),
                 '--out',report+'-audit.json'],out/(match['id']+'-audit.log'))
            summary=json.loads((folder/'cdca-summary.json').read_text())
            audit=json.loads((ROOT/(report+'-audit.json')).read_text())
            state['completed'].append({'id':match['id'],'result':summary,'audit':audit});save()
            if not summary['valid_technical_run'] or audit['technical_games'] or audit['clock_ambiguous_moves_from_rounding']:
                state.update(status='stopped_unclean_run');save();return
            if match['id']=='development' and summary['score']<=.5:
                state.update(status='stopped_no_development_advantage');save();return
        verify();state.update(status='completed');save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
