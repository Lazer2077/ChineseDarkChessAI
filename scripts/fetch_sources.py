"""Fetch fixed public sources without executing them or altering existing checkouts."""
import argparse, json, re, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('ids',nargs='+',help='IDs from docs/opponents.json');a=p.parse_args()
entries={i['id']:i for i in json.loads((root/'docs/opponents.json').read_text())['opponents']}
for key in a.ids:
    item=entries.get(key)
    if not item or not re.fullmatch('[a-zA-Z0-9_-]+',key):raise SystemExit(f'Unknown source: {key}')
    commit=item.get('commit');url=item.get('repository_url',item['url'])
    if not commit or not re.fullmatch('[a-f0-9]{40}',commit):raise SystemExit(f'No fixed commit for {key}')
    if not re.fullmatch(r'https://github.com/[\w.-]+/[\w.-]+',url):raise SystemExit(f'Non-repository URL: {url}')
    dest=root/'external'/key
    if dest.exists():
        actual=subprocess.check_output(['git','-C',dest,'rev-parse','HEAD'],text=True).strip()
        if actual!=commit or subprocess.check_output(['git','-C',dest,'status','--porcelain','--untracked-files=no'],text=True).strip():raise SystemExit(f'Existing source differs; refusing to overwrite {dest}')
        print(f'{key}: verified {commit}');continue
    dest.mkdir(parents=True)
    for args in [['git','init',str(dest)],['git','-C',str(dest),'remote','add','origin',url+'.git'],['git','-C',str(dest),'fetch','--depth','1','origin',commit],['git','-C',str(dest),'checkout','--detach','FETCH_HEAD']]:subprocess.run(args,check=True)
