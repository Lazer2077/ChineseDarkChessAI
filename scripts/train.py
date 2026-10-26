"""Search-policy distillation + terminal value regression; not equilibrium learning."""
import argparse, json, pathlib, random
import torch
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cpu');a=p.parse_args()
random.seed(a.seed);torch.manual_seed(a.seed)
rows=[]
for f in sorted(pathlib.Path(a.data).glob('game-*.json')):
    rows.extend(r for r in json.loads(f.read_text())['records'] if 'pi' in r)
if not rows: raise SystemExit('No search training rows')
w=torch.nn.Parameter(torch.zeros(15,device=a.device));v=torch.nn.Parameter(torch.zeros(15,device=a.device));opt=torch.optim.Adam([w,v],lr=.01)
for epoch in range(a.epochs):
    random.shuffle(rows);total=0.
    for r in rows:
        x=torch.tensor(r['features'],device=a.device,dtype=torch.float32);pi=torch.tensor(r['pi'],device=a.device,dtype=torch.float32)
        loss=-(pi*torch.log_softmax(x@w,dim=0)).sum()
        if r['value'] is not None:
            state=(pi[:,None]*x).sum(0);loss=loss+(torch.tanh(state@v)-r['value'])**2
        opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())
    print(json.dumps({'epoch':epoch,'loss':total/len(rows),'rows':len(rows)}),flush=True)
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'version':'action-features-v1','policy':w.detach().cpu().tolist(),'value':v.detach().cpu().tolist(),'seed':a.seed,'rows':len(rows),'epochs':a.epochs,'note':'Value head diagnostic only; search uses material cutoff.'},indent=2))
