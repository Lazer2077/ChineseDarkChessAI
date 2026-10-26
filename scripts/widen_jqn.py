"""Function-preserving capacity control: grow the attributed NNUE without losing its initial value function."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--king-buckets',type=int,choices=[1,3,9],default=1);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi/engine/train/train.py'
spec=importlib.util.spec_from_file_location('backbone_trainer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(a.seed);old=m.Net(256,16,32);old.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));old.eval();m.set_king_buckets(a.king_buckets);large=m.Net(1024,64,64);large.eval()
with torch.no_grad():
    for bucket in range(a.king_buckets):
        lo=bucket*1440
        large.ft.weight[lo:lo+1440,:256]=old.ft.weight[:1440,:256];large.ft.weight[lo:lo+1440,1024]=old.ft.weight[:1440,256]
    large.ft.weight[m.NUM_PIECE_FEATURES:m.NUM_FEATURES,:256]=old.ft.weight[1440:1470,:256];large.ft.weight[m.NUM_PIECE_FEATURES:m.NUM_FEATURES,1024]=old.ft.weight[1440:1470,256]
    large.ft_bias[:256]=old.ft_bias[:256];large.ft_bias[1024]=old.ft_bias[256]
    large.l1.weight[:16].zero_();large.l1.weight[:16,:256]=old.l1.weight[:,:256];large.l1.weight[:16,1024:1280]=old.l1.weight[:,256:512];large.l1.bias[:16]=old.l1.bias
    large.l2.weight[:32].zero_();large.l2.weight[:32,:16]=old.l2.weight;large.l2.bias[:32]=old.l2.bias
    large.out.weight.zero_();large.out.weight[:,:32]=old.out.weight;large.out.bias.copy_(old.out.bias)
    # Padding must be neutral for every newly introduced feature channel.
    large.ft.weight[m.PAD].zero_()
    us=torch.randint(0,1471,(256,48));them=torch.randint(0,1471,(256,48))
    bucket=(torch.arange(256)%a.king_buckets)[:,None]
    def remap(x):return torch.where(x<1440,x+bucket*1440,torch.where(x<1470,x-1440+m.NUM_PIECE_FEATURES,m.PAD))
    delta=float((old(us,them)-large(remap(us),remap(them))).abs().max())
    if delta>2e-6:raise RuntimeError(f'Initial value function differs by {delta}')
# Extra units must receive a learning signal, rather than remaining disconnected.
large.zero_grad();probe=large(remap(us),remap(them)).square().mean();probe.backward()
new_unit_gradient=float(large.out.weight.grad[:,32:].abs().sum())
if not new_unit_gradient>0:raise RuntimeError('Expanded units have no initial learning signal')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.save(large.state_dict(),out/'initial.pt');large.export(out/'initial.jqn')
manifest={'source_checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'initial_net_sha256':hashlib.sha256((out/'initial.jqn').read_bytes()).hexdigest(),'small_parameters':sum(p.numel() for p in old.parameters()),'large_parameters':sum(p.numel() for p in large.parameters()),'tested_feature_pairs':256,'max_initial_output_difference':delta,'dimensions':[m.NUM_FEATURES,1024,64,64],'king_buckets':a.king_buckets,'seed':a.seed,'new_output_unit_gradient_l1':new_unit_gradient,'attribution':'Upstream jqe architecture and teacher; our capacity expansion preserves the initial function. Not a learned or promoted candidate.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
