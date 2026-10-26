"""Run the pinned external referee with explicitly selected protocol padding.

The classical Pikafish parser requires x padding on public reveals. Strict
adapters/AB-JChess reject that padding. This changes serialization only.
"""
import argparse, importlib.util, json, sys, threading, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--strict-name',action='append',default=[]);p.add_argument('--timing-log');p.add_argument('--warmup',action='store_true');a,rest=p.parse_known_args()
root=Path(__file__).resolve().parents[1];path=root/'external/devinokeefe-jieqi/engine/tools/match.py'
spec=importlib.util.spec_from_file_location('cdca_external_referee',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.parse_engine
def parse_engine(specification):
    config=original(specification)
    if config['name'] in a.strict_name:config['pad']=False
    return config
module.parse_engine=parse_engine
lock=threading.Lock()
original_go=module.Engine.go
original_init=module.Engine.__init__
def timed_go(self,events,go_args,timeout):
    started=time.monotonic_ns()
    try:return original_go(self,events,go_args,timeout)
    finally:
        if a.timing_log:
            record={'engine':self.name,'pid':self.proc.pid,'ply':len(events),'go':go_args,'elapsed_ms':(time.monotonic_ns()-started)/1e6,'warmup':getattr(self,'cdca_warming',False)}
            with lock:
                with open(a.timing_log,'a') as f:f.write(json.dumps(record)+'\n')
def warmed_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    if a.warmup:
        self.cdca_warming=True
        try:self.go([],'movetime 1',60)
        except BaseException:
            self.close();raise
        finally:self.cdca_warming=False
module.Engine.go=timed_go
module.Engine.__init__=warmed_init
sys.argv=[str(path),*rest]
module.main()
