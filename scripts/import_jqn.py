"""Losslessly import the published JQN1 v1 float net for attributed warm-start research."""
import argparse, hashlib, importlib.util, json, struct
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--net',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi/engine/train/train.py'
spec=importlib.util.spec_from_file_location('cdca_backbone_trainer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw=Path(a.net).read_bytes()
if raw[:4]!=b'JQN1':raise SystemExit('Invalid JQN signature')
version,features,ft,l1,l2,scale=struct.unpack('<5If',raw[4:28])
if version!=1 or features!=1470 or scale!=400:raise SystemExit('Only the pinned v1/1470/scale400 architecture is supported')
net=m.Net(ft,l1,l2,out_scale=scale);state=net.state_dict();pos=28
for key in ['ft.weight','ft_bias','l1.weight','l1.bias','l2.weight','l2.bias','out.weight','out.bias']:
    shape=state[key].shape
    saved_shape=(features,ft+1) if key=='ft.weight' else shape
    count=int(np.prod(saved_shape));values=np.frombuffer(raw,dtype='<f4',count=count,offset=pos).copy().reshape(saved_shape);pos+=count*4
    if not np.isfinite(values).all():raise SystemExit('Non-finite network weights')
    if key=='ft.weight':state[key].zero_();state[key][:features].copy_(torch.from_numpy(values))
    else:state[key].copy_(torch.from_numpy(values))
if pos!=len(raw):raise SystemExit('Unexpected trailing network data')
net.load_state_dict(state);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
net.export(out/'roundtrip.jqn')
if (out/'roundtrip.jqn').read_bytes()!=raw:raise SystemExit('Import/export roundtrip differs')
torch.save(state,out/'warm-start.pt')
(out/'manifest.json').write_text(json.dumps({'source_net':str(Path(a.net).resolve()),'source_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_byte_identical':True,'architecture':{'features':features,'ft':ft,'l1':l1,'l2':l2,'scale':scale},'attribution':'Unchanged upstream jqe network imported for warm-start research, not our trained contribution.'},indent=2)+'\n')
print('Warm-start conversion passed byte-identical roundtrip',flush=True)
