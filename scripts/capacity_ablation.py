"""Dependent same-data NNUE capacity experiment; never consumes partial actor output."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--out',required=True);p.add_argument('--referee-python',required=True);p.add_argument('--physical-gpu',type=int,default=4);a=p.parse_args()
source=Path(a.source_run).resolve();out=Path(a.out).resolve()
if out.exists():raise SystemExit('Use a new output directory')
out.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(state):
 tmp=out/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
state={'status':'waiting_for_complete_generation','pid':os.getpid(),'source_run':str(source),'args':vars(a),'commands':[],'note':'Attribution: jqe native architecture/trainer. Function-preserving capacity growth is project code. No strength claim.'};save(state)
marker=source/'iter-001.complete.json'
while not marker.exists():
 parent=json.loads((source/'state.json').read_text());pid=parent['controller_pid'];cmd=Path(f'/proc/{pid}/cmdline')
 if not cmd.exists() or b'continual_learning.py' not in cmd.read_bytes():raise SystemExit('Source controller is not live and generation is not finalized; no data admitted')
 state['verified_source_controller_pid']=pid;save(state);time.sleep(15)
attestation=json.loads(marker.read_text());data=source/'iter-001.bin';val=source/'validation.bin';valmeta=json.loads((source/'validation.complete.json').read_text())
if sha(data)!=attestation['sha256'] or sha(val)!=valmeta['sha256']:raise SystemExit('Input data digest mismatch')
if attestation['recipe']['games']!=8192 or attestation['recipe']['nodes']!=5000:raise SystemExit('Unexpected capacity-control dataset')
state.update(status='inputs_verified',training_records=attestation['records'],validation_records=valmeta['records'],training_sha256=sha(data),validation_sha256=sha(val));save(state)
# This keeps batch learning separate from the original controller's GPU 2 learner.
os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu);os.environ['OMP_NUM_THREADS']='2'
while True:
 free=int(subprocess.check_output(['nvidia-smi',f'--id={a.physical_gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free>=4096:break
 state.update(status='waiting_for_gpu_memory',gpu_free_mib=free);save(state);time.sleep(30)
def run(stage,command):
 state.update(status='running',stage=stage);state['commands'].append(list(map(str,command)));save(state)
 with open(out/f'{stage}.log','w') as log:
  child=subprocess.Popen(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,cwd=r);state['child_pid']=child.pid;save(state);code=child.wait()
 state['child_pid']=None;save(state)
 if code:raise RuntimeError(f'{stage} failed with code {code}')
try:
 teacher=r/'external/devinokeefe-jieqi/engine/nets/it014.jqn';backbone=r/'external/devinokeefe-jieqi/engine';init=source/'initial-import/warm-start.pt'
 for label,kb in [('wide',1),('wide-king',9)]:run(label+'-initialize',[sys.executable,r/'scripts/widen_jqn.py','--checkpoint',init,'--out',out/(label+'-initial'),'--king-buckets',kb,'--seed',1])
 models={}
 for label in ['small','wide','wide-king']:
  checkpoint=init if label=='small' else out/(label+'-initial')/'initial.pt'
  dims=[] if label=='small' else ['--ft',1024,'--l1',64,'--l2',64,'--kb',9 if label=='wide-king' else 1]
  run(label+'-train',[sys.executable,backbone/'train/train.py',data,'--val-file',val,'--out',out/label,'--init',checkpoint,*dims,'--epochs',15,'--batch',4096,'--lr',.00002,'--lam',.7,'--k',430,'--seed',20261027,'--data-on-gpu'])
  models[label]=sha(out/label/'best.jqn');state['model_sha256']=models;save(state)
 for label,seed in [('wide',7500001),('wide-king',7600001)]:
  run(label+'-match',[a.referee_python,r/'scripts/native_match.py','--candidate',backbone/'build/jqe','--baseline',backbone/'build/jqe','--candidate-net',out/label/'best.jqn','--net',out/'small/best.jqn','--tc','5+0.05','--games',64,'--seed',seed,'--out',out/(label+'-gate')])
 state.update(status='completed',stage='finished');save(state)
except BaseException as error:state.update(status='failed',error=str(error));save(state);raise
