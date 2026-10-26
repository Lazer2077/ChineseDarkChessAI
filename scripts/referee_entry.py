"""Run the pinned external referee with explicitly selected protocol padding.

The classical Pikafish parser requires x padding on public reveals. Strict
adapters/AB-JChess reject that padding. This changes serialization only.
"""
import argparse, importlib.util, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--strict-name',action='append',default=[]);a,rest=p.parse_known_args()
root=Path(__file__).resolve().parents[1];path=root/'external/devinokeefe-jieqi/engine/tools/match.py'
spec=importlib.util.spec_from_file_location('cdca_external_referee',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.parse_engine
def parse_engine(specification):
    config=original(specification)
    if config['name'] in a.strict_name:config['pad']=False
    return config
module.parse_engine=parse_engine
sys.argv=[str(path),*rest]
module.main()
