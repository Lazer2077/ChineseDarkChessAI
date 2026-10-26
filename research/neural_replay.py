"""Observer-only, replay-certified trajectories for neural search self-play.

Search is the attributed root-information PUCT; environment RNG and search RNG
are separate. Recorded rule outcomes are Yuan tiantian-v5, not native-jqe wins.
"""
import hashlib,json,random
from pathlib import Path
import numpy as np
from research.policy_value import AZ,CONTRACT,REC,records_from_info

SCHEMA='cdca-neural-search-replay-v1'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def fresh_game(seed):
    rng=random.Random(seed)
    state=AZ.jq.GameState.initial(rng=rng)
    return state,AZ.im.init_belief_for_game(state),rng

def episode(engine,seed,simulations,max_plies=1024):
    state,belief,environment_rng=fresh_game(seed)
    history=[];raw=[];counts=[];actions=[];reveals=[];roots=[];sides=[]
    engine.deadline=None;engine.rng.seed(seed^0x5A17C9)
    while state.game_outcome()[0] is None:
        if len(history)>=max_plies:raise ValueError('Resource-truncated episode; no invented terminal label')
        info=belief.info_state(state,state.side)
        policy,move,root=engine.search(state,belief,history,
            temperature=1.0 if len(history)<30 else .1,simulations=simulations,
            add_dirichlet=True,dirichlet_alpha=.3,dirichlet_epsilon=.25,return_root=True)
        legal=state.legal_moves()
        if move is None or move not in legal or root.visits!=simulations:
            raise ValueError('Incomplete search or illegal selected move')
        visits=np.zeros(AZ.MOVE_SIZE,dtype=np.uint32)
        for choice,child in root.children.items():visits[AZ.move_to_index(choice)]=child.visits
        if visits.sum()!=root.visits or not np.allclose(np.asarray(policy),visits/root.visits,atol=1e-7):
            raise ValueError('Policy target does not match actual completed visits')
        raw.append(records_from_info(info)[0]);counts.append(visits);roots.append(root.visits);sides.append(state.side)
        undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=environment_rng)
        history.append(move);actions.append(move);reveals.append((undo.moved_reveal,undo.captured_reveal))
    outcome,reason=state.game_outcome()
    result={'records':np.asarray(raw,dtype=np.uint8),'visit_counts':np.asarray(counts,dtype=np.uint32),
        'root_visits':np.asarray(roots,dtype=np.uint32),'sides':np.asarray(sides,dtype=np.uint8),
        'moves':np.asarray(actions,dtype=np.uint8),'reveals':np.asarray(reveals,dtype=np.uint8),
        'values':np.asarray([AZ.outcome_value_for_player(outcome,s) for s in sides],dtype=np.float32)}
    metadata={'schema':SCHEMA,'input_contract':CONTRACT,'seed':seed,'simulations':simulations,
        'positions':len(history),'outcome':outcome,'reason':reason,
        'rules':'Original pinned Yuan tiantian-v5; full history and captor-private belief updates retained. External native-jqe benchmark profile differs.',
        'environment_rng':'Independent random.Random(seed), lazy uniform without-replacement reveals/captures; never shared with search RNG.',
        'scope':'Current neural model searches both sides. Raw normalized completed root visits and terminal actor-relative values; no teacher one-hot policy or teacher bootstrap value.'}
    audit_episode(result,metadata)
    return result,metadata

def audit_episode(data,metadata):
    """Replay every real move and both private ledgers; do not expose them to search."""
    n=metadata['positions']
    shapes={'records':(n,112),'visit_counts':(n,AZ.MOVE_SIZE),'root_visits':(n,),
        'sides':(n,),'moves':(n,2),'reveals':(n,2),'values':(n,)}
    dtypes={'records':'uint8','visit_counts':'uint32','root_visits':'uint32','sides':'uint8','moves':'uint8','reveals':'uint8','values':'float32'}
    if metadata['schema']!=SCHEMA or metadata['input_contract']!=CONTRACT or not n or set(data)!=set(shapes):raise ValueError('Replay schema differs')
    for key,shape in shapes.items():
        if data[key].shape!=shape or str(data[key].dtype)!=dtypes[key] or not np.isfinite(data[key]).all():raise ValueError('Invalid replay array: '+key)
    state,belief,rng=fresh_game(metadata['seed'])
    full=0;private_captures=0
    for i in range(n):
        if state.game_outcome()[0] is not None:raise ValueError('Record after terminal')
        if state.side!=int(data['sides'][i]):raise ValueError('Wrong actor')
        expected=records_from_info(belief.info_state(state,state.side))[0]
        if not np.array_equal(expected,data['records'][i]):raise ValueError('Observation differs from complete-history replay')
        rec=expected.reshape(1,112).view(REC)
        full+=not ((rec['board']&15)==8).any()
        legal=state.legal_moves();legal_ids=np.asarray([AZ.move_to_index(m) for m in legal],dtype=np.int64)
        counts=data['visit_counts'][i];root=int(data['root_visits'][i]);chosen=tuple(map(int,data['moves'][i]))
        if root!=metadata['simulations'] or int(counts.sum())!=root or root<=0 or int(counts[legal_ids].sum())!=root or chosen not in legal or not counts[AZ.move_to_index(chosen)]:
            raise ValueError('Invalid completed visits, policy support or selected action')
        undo=AZ.im.apply_move_and_update_belief(state,belief,chosen,rng=rng)
        if tuple(map(int,data['reveals'][i]))!=(undo.moved_reveal,undo.captured_reveal):raise ValueError('Environment seed/reveal mismatch')
        private_captures+=bool(undo.captured_reveal)
    outcome,reason=state.game_outcome()
    if outcome is None or outcome!=metadata['outcome'] or reason!=metadata['reason']:raise ValueError('Incomplete or incorrect rule outcome')
    targets=np.asarray([AZ.outcome_value_for_player(outcome,int(s)) for s in data['sides']],dtype=np.float32)
    if not np.array_equal(targets,data['values']):raise ValueError('Wrong terminal value perspective')
    return {'positions_replayed':n,'fully_public_positions':int(full),'private_dark_captures':int(private_captures),
        'legal_policy_support':True,'root_visit_counts_match':True,'observer_records_match':True,'terminal_values_match':True}

def save_episode(folder,index,data,metadata):
    folder=Path(folder);base=folder/f'game-{index:06d}';target=base.with_suffix('.npz')
    if target.exists() or base.with_suffix('.json').exists():raise ValueError('Refuse to overwrite an episode')
    with base.with_suffix('.npz.tmp').open('wb') as handle:np.savez_compressed(handle,**data)
    base.with_suffix('.npz.tmp').replace(target)
    metadata={**metadata,'npz_sha256':sha(target),'audit':audit_episode(data,metadata)}
    save_json(base.with_suffix('.json'),metadata)
    return metadata

def load_episode(path):
    path=Path(path);metadata=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=metadata['npz_sha256']:raise ValueError('Episode digest differs')
    with np.load(path,allow_pickle=False) as source:data={key:source[key] for key in source.files}
    audit_episode(data,metadata)
    return data,metadata
