"""Public-continuation decision sharing, with complete public rule memory.

Only the no-on-board-dark phase may forget private capture paths/dead bags.
Public path, repetition counts, check/chase state and rules remain in the key.
This is a perfect-information continuation abstraction, not a model of arbitrary
opponent policies that condition their behavior on dead private information.
"""
import dataclasses,hashlib,inspect,textwrap,time
from research.neural_search import ObservationPUCT
from research.policy_value import AZ

def public_continuation_key(state,public_path):
    if any(AZ.jq.is_dark(p) for p in state.board):return None
    if getattr(state,'repetition_history_complete',True) is False or state.position_counts is None:return None
    return ('cdca-public-continuation-v1',tuple(public_path),state.side,bytes(state.board),
        state.no_capture_plies,tuple(state.consecutive_checks),tuple(state.check_anchor),
        tuple(state.chase_streak),tuple(state.chase_target),tuple(state.chase_target_code),
        tuple(state.chase_anchor),tuple(sorted(state.position_counts.items())),
        dataclasses.astuple(state.rules))

class PublicContinuationPUCT(ObservationPUCT):
    def __init__(self,*args,share_public=True,**kwargs):
        super().__init__(*args,**kwargs);self.share_public=share_public

    def _reset_graph_stats(self):
        self._full_equivalence={};self._full_requests=0;self._partial_requests=0

    def _key(self,state,public_path,original):
        public=public_continuation_key(state,public_path)
        if public is None:self._partial_requests+=1;return original
        self._full_requests+=1
        self._full_equivalence.setdefault(public,set()).add(original)
        return public if self.share_public else original

    def _root_key(self,info):
        return self._key(info,(),AZ.im.information_set_key(info,(),()))

    def _decision_key(self,belief,state,observer,public_path,private_path):
        original=belief.information_set_key(state,observer,public_path,private_path)
        return self._key(state,public_path,original)

    def graph_statistics(self):
        original=sum(len(group) for group in self._full_equivalence.values())
        return {'share_public':self.share_public,'partial_key_requests':self._partial_requests,
            'full_key_requests':self._full_requests,'distinct_original_full_keys':original,
            'public_continuation_classes':len(self._full_equivalence),
            'removable_private_context_duplicates':original-len(self._full_equivalence)}

def _install():
    source=textwrap.dedent(inspect.getsource(AZ.NeuralMCTS.search))
    replacements={
        '    sims = simulations or self.config.mcts_simulations\n':'    self._reset_graph_stats()\n    sims = simulations or self.config.mcts_simulations\n',
        'im.information_set_key(root_info, (), ())':'self._root_key(root_info)',
        'sim_view.information_set_key(\n':'self._decision_key(sim_view,\n',
        '    while completed < sims:\n':'    while completed < sims:\n        if completed and self.deadline is not None and _cdca_clock() >= self.deadline:\n            break\n'}
    for needle,replacement in replacements.items():
        if source.count(needle)!=1:raise ValueError('Pinned PUCT structure changed: '+needle)
        source=source.replace(needle,replacement)
    namespace=dict(AZ.__dict__,_cdca_clock=time.perf_counter)
    exec(compile(source,'<cdca-public-continuation-puct>','exec'),namespace)
    PublicContinuationPUCT.search=namespace['search']
    return hashlib.sha256(source.encode()).hexdigest()

SEARCH_SHA256=_install()
