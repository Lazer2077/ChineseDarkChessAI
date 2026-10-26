"""Measure FP32 forward costs separately from full-search equal-clock strength."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.neural_search import load_model
from research.policy_value import REC,encode_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--records',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('Refuse to overwrite timing evidence')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    raw=np.fromfile(a.records,dtype=REC,count=32).view(np.uint8).reshape(-1,112)
    x=encode_records(torch.from_numpy(raw).cuda());rows=[]
    for directory in a.models:
        net,metadata=load_model(Path(directory),'cuda')
        for batch in [1,32]:
            inputs=x[:batch];device_times=[];wall_times=[]
            with torch.inference_mode():
                for _ in range(10):net(inputs)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for _ in range(50):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter();begin.record();net(inputs);end.record();end.synchronize()
                    wall_times.append(1000*(time.perf_counter()-start));device_times.append(begin.elapsed_time(end))
            rows.append({'model':directory,'model_sha256':hashlib.sha256((Path(directory)/'best.npz').read_bytes()).hexdigest(),
                'channels':metadata['channels'],'blocks':metadata['blocks'],'batch':batch,
                'wall_median_ms':float(np.median(wall_times)),'wall_p95_ms':float(np.quantile(wall_times,.95)),
                'device_median_ms':float(np.median(device_times)),
                'device_p95_ms':float(np.quantile(device_times,.95)),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        del net;torch.cuda.empty_cache()
    report={'schema':1,'status':'completed','scope':'Forward-only timing, excludes encoder, belief construction, tree search and game-clock overhead. Sequential model order, not a randomized speedup estimate.',
        'precision':'FP32, no autocast (matching search actor)','warmups':10,'repeats':50,
        'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,
        'gpu_model':torch.cuda.get_device_name(),
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'records_sha256':hashlib.sha256(Path(a.records).read_bytes()).hexdigest(),'measurements':rows}
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
