"""Canonicalize irrelevant off-board inventories in full-public training states only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=Path(a.data);out=Path(a.out);meta=json.loads(source.with_suffix('.complete.json').read_text())
if source.suffix!='.bin' or source.stat().st_size%112 or hashlib.sha256(source.read_bytes()).hexdigest()!=meta['sha256'] or meta['rejected_games']:raise SystemExit('Require finalized valid source')
if out.exists():raise SystemExit('Do not overwrite data')
raw=np.fromfile(source,dtype=np.uint8).reshape(-1,112);full=((raw[:,:90]&15)==8).sum(1)==0;changed=full&(raw[:,91:103].sum(1)>0)
raw[full,91:103]=0
out.parent.mkdir(parents=True,exist_ok=True);raw.tofile(out)
record={'source_sha256':meta['sha256'],'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'records':len(raw),'full_public_records':int(full.sum()),'records_changed':int(changed.sum()),'change':'Only urn columns 91:103 set to zero when no on-board DARK remains. Original search-score and outcome labels, flags and all other fields unchanged.'}
out.with_suffix('.canonical.json').write_text(json.dumps(record,indent=2)+'\n')
