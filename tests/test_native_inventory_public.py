import subprocess
import sys


def test_private_capture_fields_and_seed_cannot_change_public_frames():
    # Native referee and Yuan modules share the import name jieqi; isolate this
    # test so collecting neural-search tests cannot change its implementation.
    program = '''
import copy,gzip,json
from pathlib import Path
from scripts.calibrate_native_inventory import public_events,public_frames,NativeRules,ROOT
g=json.loads(gzip.decompress((ROOT/'results/raw/teacher-a-6a1addf2665a.jsonl.gz').read_bytes()).splitlines()[0])
other=copy.deepcopy(g);other['seed']='123'
for row in other['moves']:row['c']='P'
rules=NativeRules(ROOT/'external/devinokeefe-jieqi/build/release/lib/libjieqi_rules.so')
assert public_events(g)==public_events(other)
first=list(public_frames(public_events(g),rules));second=list(public_frames(public_events(other),rules))
assert first==second and len(first)==g['plies']
'''
    subprocess.run([sys.executable, '-c', program], check=True, timeout=30)
