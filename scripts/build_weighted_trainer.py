"""Make a pinned local trainer copy with optional aligned per-record loss weights."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi';out=Path(a.out)
if out.exists():raise SystemExit('Use fresh output')
if subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',source,'diff','HEAD'],text=True).strip():
 raise SystemExit('Require pinned unchanged MIT trainer')
original=(source/'engine/train/train.py').read_text();s=original
changes=[
('    args = ap.parse_args()', '''    ap.add_argument('--weight-file', required=True)
    args = ap.parse_args()
    if args.stream or args.max_records or not args.val_file or not args.data_on_gpu:
        ap.error('Weighted ablation requires full in-GPU replay and separate validation')'''),
('    _, _, _, s_all, r_all, _ = fields(','''    weight_array = np.fromfile(args.weight_file, dtype='<f4')
    if len(weight_array) != len(train) or not np.isfinite(weight_array).all() or (weight_array <= 0).any() or abs(float(weight_array.mean()) - 1) > 1e-5:
        raise ValueError('Require positive mean-one weights aligned to filtered records')
    train_weights = torch.from_numpy(weight_array).to(dev)
    _, _, _, s_all, r_all, _ = fields('''),
('    def batch_loss(b, mirror=True):','    def batch_loss(b, mirror=True, weights=None):'),
('        return ((pred - target) ** 2).mean(), pred, result', '''        errors = (pred - target) ** 2
        return (errors if weights is None else errors * weights).mean(), pred, result'''),
('            loss, _, _ = batch_loss(b)', '            weights = train_weights[perm[i * args.batch:(i + 1) * args.batch]]\n            loss, _, _ = batch_loss(b, weights=weights)')]
for old,new in changes:
 if s.count(old)!=1:raise SystemExit('Trainer patch context changed')
 s=s.replace(old,new)
compile(s,'weighted-train.py','exec')
out.mkdir(parents=True);(out/'train.py').write_text(s);(out/'LICENSE.jqe').write_bytes((source/'LICENSE').read_bytes())
manifest={'backbone_commit':'331a6e612a038c4a04cf980f7b9482910fd50078','upstream_trainer_sha256':hashlib.sha256(original.encode()).hexdigest(),
 'weighted_trainer_sha256':hashlib.sha256(s.encode()).hexdigest(),'change':'Mean of per-record squared errors times fixed mean-one replay weights; validation remains unweighted. Same shuffle, mirror augmentation, architecture, labels, optimizer, scheduler, clipping and export. MIT attribution retained.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
