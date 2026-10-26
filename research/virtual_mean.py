"""Virtual-mean pending evaluations for the attributed information-set PUCT.

Mechanism follows Cazenave, Batch Monte Carlo Tree Search (2104.04278). This
adaptation preserves chance-chain player perspectives and complete observer
history; it does not transfer a perfect-information policy-improvement guarantee.
"""
import hashlib,inspect,re,textwrap,time
from research.policy_value import AZ
from research.neural_search import ObservationPUCT


class CreditPath(list):
    """Track each path's temporary credits, including same-player chance layers."""
    def __init__(self,root):
        super().__init__([root]);self.credits=[0.];self.reserved={};self.finalized=False

    def reserve(self,node):
        if self.finalized or id(node) in self.reserved:
            raise ValueError('Duplicate/unbacked reservation')
        credit=node.value_sum/node.visits if node.visits else 0.
        node.visits+=1;node.value_sum+=credit
        self.reserved[id(node)]=credit

    def append(self,node):
        if self.finalized or id(node) not in self.reserved:
            raise ValueError('Path node lacks its reservation')
        self.credits.append(self.reserved.pop(id(node)));super().append(node)

    def extend(self,nodes):
        for node in nodes:self.append(node)


class VirtualMeanPUCT(ObservationPUCT):
    @staticmethod
    def _backup(path,value,virtual_loss):
        if not isinstance(path,CreditPath) or path.finalized or path.reserved:
            raise ValueError('Incomplete or duplicate credit backup')
        for i in range(len(path)-1,-1,-1):
            node=path[i]
            if i==0:
                node.visits+=1;node.value_sum+=value
            else:
                node.value_sum+=value-path.credits[i]
                if path[i-1].player_to_move!=node.player_to_move:value=-value
        path.finalized=True


def _install():
    source=textwrap.dedent(inspect.getsource(AZ.NeuralMCTS.search))
    source,count=re.subn(r'(?m)^(?P<indent> +)path = \[node\]$',
        lambda m:m['indent']+'path = _CreditPath(node)',source)
    if count!=1:raise ValueError('Pinned path initialization changed')
    pattern=r'(?P<indent> +)(?P<node>child|chance_node|node)\.visits \+= 1\n(?P=indent)(?P=node)\.value_sum \+= virtual_loss'
    source,count=re.subn(pattern,lambda m:m['indent']+'path.reserve('+m['node']+')',source)
    if count!=4:raise ValueError('Pinned reservation sites changed')
    needle='    while completed < sims:\n'
    if source.count(needle)!=1:raise ValueError('Pinned batch loop changed')
    source=source.replace(needle,needle+'''        if completed and self.deadline is not None and _cdca_clock() >= self.deadline:
            break
''')
    namespace=dict(AZ.__dict__,_CreditPath=CreditPath,_cdca_clock=time.perf_counter)
    exec(compile(source,'<cdca-virtual-mean-search>','exec'),namespace)
    VirtualMeanPUCT.search=namespace['search']
    return hashlib.sha256(source.encode()).hexdigest()


SEARCH_SHA256=_install()
