"""Observer-only bridge to pinned Yuan-Bai/jieqi-lab; no opponent source edits.

Inputs are UCI coordinates and this viewer's reveal/capture suffixes, never a
true deal. Unknown past captured types are not invented as player knowledge.
"""
import hashlib,importlib.util,json,os,random,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/yuan-bai-jieqi-lab'
PINS={'jieqi.py':'315ab6c37513bdedf11b692309ef15e92cc8c4261220e9f64a6752fd85a50230',
      'ismcts.py':'e4f44974863a35c5efa5f85877b981a954731fdf4362feaea1ca70763dd42d73'}
def load_sources():
    for name,digest in PINS.items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Pinned Yuan source changed')
    native_dir=os.environ.get('YUAN_NATIVE_DIR')
    if native_dir:
        directory=Path(native_dir).resolve()
        manifest=json.loads((directory.parent/'manifest.json').read_text())
        if manifest['commit']!='4173aa3ddc72e19227a619ffc4ea80a88b3ba2b4':raise ValueError('Wrong native source revision')
        for filename,digest in manifest['binary_sha256'].items():
            if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Native binary changed')
        if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    # jieqi-lab's ismcts imports the name jieqi. A dedicated process is required
    # when also using the unrelated native-referee package with that name.
    if 'jieqi' in sys.modules and Path(sys.modules['jieqi'].__file__).resolve()!=SOURCE/'jieqi.py':
        raise RuntimeError('Use an isolated process for the Yuan rules module')
    for name in ['jieqi','ismcts']:
        if name in sys.modules:continue
        spec=importlib.util.spec_from_file_location(name,SOURCE/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules['jieqi'],sys.modules['ismcts']

def reconstruct(tokens,viewer=None,placeholder_seed=0):
    jq,mc=load_sources()
    actor=jq.RED if len(tokens)%2==0 else jq.BLACK
    if viewer is None:viewer=actor
    if viewer not in (jq.RED,jq.BLACK):raise ValueError('Invalid observer')
    state=jq.GameState.initial(rng=random.Random(placeholder_seed))
    public=[jq.POOL_INIT.copy(),jq.POOL_INIT.copy()]
    known=[row.copy() for row in public];unknown=[0,0]
    letters={'R':jq.TYPE_ROOK,'N':jq.TYPE_KNIGHT,'B':jq.TYPE_ELEPHANT,
             'A':jq.TYPE_ADVISOR,'C':jq.TYPE_CANNON,'P':jq.TYPE_PAWN}
    for token in tokens:
        if len(token)<4:raise ValueError('Malformed move')
        # Strict coordinates: negative indices must never silently index a board.
        if token[0] not in 'abcdefghi' or token[2] not in 'abcdefghi' or token[1] not in '0123456789' or token[3] not in '0123456789':
            raise ValueError('Invalid square')
        frm,to=jq.name_to_square(token[:2]),jq.name_to_square(token[2:4])
        moved,victim=state.board[frm],state.board[to];side=state.side
        if not moved or jq.piece_color(moved)!=side:raise ValueError('Wrong public mover')
        suffix=token[4:];reveal=capture=None
        def identity(ch,color):
            if ch.upper() not in letters or ch.isupper()!=(color==jq.RED):raise ValueError('Invalid identity suffix')
            return letters[ch.upper()]
        if jq.is_dark(moved):
            if not suffix:raise ValueError('Missing public reveal')
            reveal=identity(suffix[0],side);suffix=suffix[1:]
        if victim and jq.is_dark(victim):
            color=jq.piece_color(victim)
            if side==viewer:
                if not suffix:raise ValueError('Missing own private capture identity')
                capture=identity(suffix[0],color);suffix=suffix[1:]
            elif suffix:
                raise ValueError('Opponent private capture identity leaked')
        if suffix:raise ValueError('Unexpected suffix; use strict UCI tokens')
        if (frm,to) not in state._iter_legal_moves_fast(side):
            raise ValueError('Illegal public geometry or king safety')
        # Reconstruct only public geometry/rule history. apply_move requires a
        # temporary captured type; reset its urn from public counts each move so
        # this arbitrary hidden-past placeholder never constrains later reveals.
        state.pool=[row.copy() for row in public]
        state.pool_total=[sum(row) for row in public]
        state.apply_move((frm,to),reveal_choice=reveal,capture_reveal_choice=capture)
        if reveal is not None:
            public[side][reveal]-=1;known[side][reveal]-=1
        if victim and jq.is_dark(victim):
            color=jq.piece_color(victim)
            if capture is None:unknown[color]+=1
            else:known[color][capture]-=1
        if any(x<0 for row in known for x in row) or any(x<0 for row in public for x in row):
            raise ValueError('Impossible observer inventory')
    dark=[sum(bool(pc) and jq.is_dark(pc) and jq.piece_color(pc)==c for pc in state.board) for c in (jq.BLACK,jq.RED)]
    if any(sum(known[c])-unknown[c]!=dark[c] for c in (jq.BLACK,jq.RED)):
        raise ValueError('Observer inventory and public occupancy disagree')
    pools=[[row.copy() for row in public] for _ in (jq.BLACK,jq.RED)]
    pools[viewer]=[row.copy() for row in known]
    totals=[dark.copy(),dark.copy()]
    removals=[[sum(public[c])-dark[c] for c in (jq.BLACK,jq.RED)] for _ in (jq.BLACK,jq.RED)]
    removals[viewer]=unknown.copy()
    belief=mc.BeliefModel(pools,totals,removals,public)
    info=belief.info_state(state,viewer)
    # Root pool needs valid on-board sampled counts for any upstream legality
    # probe. Sampling uses only this observer's ledger and is not true-world data.
    sampled=info.determinize(random.Random(placeholder_seed))
    return sampled,belief,info

def snapshot(tokens,viewer=None,placeholder_seed=0):
    jq,_=load_sources();state,_,info=reconstruct(tokens,viewer,placeholder_seed)
    return {'side':'r' if state.side==jq.RED else 'b',
      'board':''.join('.' if not pc else (('X' if jq.is_dark(pc) else 'KRNBACP'[jq.piece_type(pc)-1]) if jq.piece_color(pc)==jq.RED else ('x' if jq.is_dark(pc) else 'KRNBACP'[jq.piece_type(pc)-1].lower())) for row in range(9,-1,-1) for pc in state.board[row*9:row*9+9]),
      'known':info.pool_known,'unknown':info.unknown_removals,'public':info.public_pool_known,
      'no_capture_plies':state.no_capture_plies,'consecutive_checks':state.consecutive_checks,
      'chase_streak':state.chase_streak,'position_counts':sorted(state.position_counts.values()),
      'rules_backend':jq.RULES_BACKEND,'legal_backend':jq.LEGAL_BACKEND,
      'legal_moves':sorted(jq.format_move_uci(move) for move in state.legal_moves())}

if __name__=='__main__':
    import json
    for line in sys.stdin:
        request=json.loads(line)
        try:reply={'ok':True,'snapshot':snapshot(**request)}
        except Exception as error:reply={'ok':False,'error':str(error)}
        print(json.dumps(reply),flush=True)
