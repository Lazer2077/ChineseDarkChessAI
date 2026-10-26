"""Generate all fixed-plan GPU neural self-play episodes; never train on truncations."""
import argparse,json,multiprocessing,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.neural_replay import episode,save_episode,save_json,sha

ENGINE=None
def initialize(model,device,leaf_batch):
    global ENGINE
    import torch
    from research.neural_search import ObservationPUCT,load_model
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net,_=load_model(ROOT/model,device)
    ENGINE=ObservationPUCT(net,device,leaf_batch=leaf_batch)

def work(task):
    index,seed,simulations,max_plies,out=task
    started=time.monotonic();data,metadata=episode(ENGINE,seed,simulations,max_plies)
    metadata['generation_seconds']=time.monotonic()-started
    return index,save_episode(out,index,data,metadata)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Use a fresh output directory')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Self-play plan changed')
        for file,expected in plan['sha256'].items():
            if sha(ROOT/file)!=expected:raise ValueError('Frozen self-play input changed: '+file)
    verify();out.mkdir(parents=True)
    status={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'completed':0,'episodes':[]}
    save_json(out/'manifest.json',status)
    try:
        tasks=[(i,plan['seed']+i,plan['simulations'],plan['max_plies'],str(out)) for i in range(plan['games'])]
        with multiprocessing.get_context('spawn').Pool(plan['workers'],initialize,(plan['model'],plan['device'],plan['leaf_batch'])) as pool:
            for index,metadata in pool.imap_unordered(work,tasks):
                verify();status['episodes'].append({'index':index,**metadata});status['completed']+=1
                save_json(out/'manifest.json',status)
                print(json.dumps({'game':index,'completed':status['completed'],'positions':metadata['positions'],'outcome':metadata['outcome']}),flush=True)
        verify();status['episodes'].sort(key=lambda row:row['index']);status['status']='completed';save_json(out/'manifest.json',status)
    except BaseException as error:
        status.update(status='failed',error=str(error));save_json(out/'manifest.json',status);raise

if __name__=='__main__':main()
