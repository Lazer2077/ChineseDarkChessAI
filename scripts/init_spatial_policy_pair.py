"""Create matched wider backbones with dense versus relative policy decoders."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json
from research.policy_value import REC,encode_records,export_npz
from research.spatial_policy_value import make_model,POLICY_MAPPING_SHA256


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Initialization plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen initialization input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    dense=make_model('dense2550',plan['channels'],plan['blocks']).eval()
    torch.manual_seed(plan['seed']);relative=make_model('relative50',plan['channels'],plan['blocks']).eval()
    d=dense.state_dict();r=relative.state_dict();common=sorted(d.keys()&r.keys())
    if not all(torch.equal(d[k],r[k]) for k in common):raise ValueError('Common model initialization differs')
    records=np.fromfile(ROOT/plan['sample_data'],dtype=REC,count=64).view(np.uint8).reshape(-1,112)
    x=encode_records(torch.from_numpy(records))
    with torch.inference_mode():pd,vd=dense(x);pr,vr=relative(x)
    if not torch.equal(vd,vr) or pd.shape!=pr.shape or pd.shape[1]!=2550:raise ValueError('Initial value/geometry differs')
    report={'schema':1,'status':'completed','plan_sha256':digest,'channels':plan['channels'],'blocks':plan['blocks'],
        'seed':plan['seed'],'policy_mapping_sha256':POLICY_MAPPING_SHA256,'common_state_arrays_exact':True,
        'common_state_array_count':len(common),'initial_value_functions_equal_on64training_rows':True,
        'initial_policy_functions_equal':bool(torch.equal(pd,pr)),'models':[],
        'scope':'Matched common trunk/value/policy-conv initialization only; classifier functions differ by design. Sampled TRAIN rows are an implementation check, not selection or strength evidence.'}
    for label,net in [('dense2550',dense),('relative50',relative)]:
        file=out/(label+'.npz');export_npz(net,file)
        report['models'].append({'policy_head':label,'parameters':sum(p.numel() for p in net.parameters()),
            'final_classifier_parameters':sum(p.numel() for p in (net.policy_fc if label=='dense2550' else net.policy_spatial).parameters()),
            'initial_npz':str(file.relative_to(ROOT)),'initial_sha256':sha(file)})
    verify();save_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
