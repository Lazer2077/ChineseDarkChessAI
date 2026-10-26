"""Pinned root-information PUCT with the CDCA observation encoder and safe arrays.

Search remains the original Yuan implementation. The sole control-flow change
checks a deadline between completed leaf batches, never during virtual-loss
selection/backup. No complete public rule history is discarded by the search.
"""
import hashlib,inspect,random,textwrap,time
from pathlib import Path
import numpy as np
import torch
from research.policy_value import AZ,CONTRACT,PhasePolicyValueNet,encode_records,records_from_info,load_npz

class ObservationPUCT(AZ.NeuralMCTS):
    def __init__(self,net,device='cpu',seed=20261029,leaf_batch=16):
        config=AZ.AZConfig(mcts_simulations=128,leaf_batch_size=leaf_batch,chance_nodes=True,chance_k=0)
        super().__init__(net,config,torch.device(device),random.Random(seed))
        self.deadline=None

    def _evaluate_batch(self,infos,histories,*,repetition_entries=None):
        # histories/entries remain in search state and decision keys. The learned
        # encoder deliberately omits these features, just as during warm start.
        raw=np.concatenate([records_from_info(info) for info in infos])
        x=encode_records(torch.from_numpy(raw).to(self.device))
        self.net.eval()
        with torch.inference_mode():
            policy,value=self.net(x)
            policies=torch.softmax(policy.float(),dim=1).cpu().numpy()
            values=value.float().cpu().tolist()
        policies.setflags(write=False)
        return policies,values

    def _evaluate(self,info_state,history):
        policy,value=self._evaluate_batch([info_state],[history])
        return torch.from_numpy(policy[0].copy()),value[0]

def _install_batch_deadline():
    source=textwrap.dedent(inspect.getsource(AZ.NeuralMCTS.search))
    needle='    while completed < sims:\n'
    if source.count(needle)!=1:raise ValueError('Pinned PUCT batch loop changed')
    patched=source.replace(needle,needle+'''        if completed and self.deadline is not None and _cdca_clock() >= self.deadline:
            break
''')
    namespace=dict(AZ.__dict__,_cdca_clock=time.perf_counter)
    exec(compile(patched,'<cdca-neural-batch-deadline>','exec'),namespace)
    ObservationPUCT.search=namespace['search']
    return hashlib.sha256(patched.encode()).hexdigest()

TIMED_SEARCH_SHA256=_install_batch_deadline()

def load_model(directory,device='cpu'):
    import json
    directory=Path(directory);metadata=json.loads((directory/'best.json').read_text())
    path=directory/'best.npz'
    if metadata['input_contract']!=CONTRACT or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['npz_sha256']:
        raise ValueError('Model input contract or array digest differs')
    net=PhasePolicyValueNet(metadata['channels'],metadata['blocks'],metadata['phase_experts'])
    load_npz(net,path);net.to(device);net.eval()
    return net,metadata
