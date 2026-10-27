"""Query the unchanged native expert with a complete lawful actor history."""
import re
from research.policy_value import AZ
from research.native_opponent_policy import FixedDepthOpponent


def observer_token(move,undo,viewer):
    if viewer not in (AZ.jq.BLACK,AZ.jq.RED):raise ValueError('Invalid observer')
    token=AZ.jq.format_move_uci(move);mover=undo.prev_side
    letters=dict(zip(AZ.jq.POOL_TYPES,'RNBACP'))
    def letter(kind,side):
        if kind not in letters:raise ValueError('Invalid observed dark identity')
        return letters[kind] if side==AZ.jq.RED else letters[kind].lower()
    if undo.moved_reveal:token+=letter(undo.moved_reveal,mover)
    # Do not even read the opponent's private identity from the event object.
    if mover==viewer and undo.captured_before and AZ.jq.is_dark(undo.captured_before):
        token+=letter(undo.captured_reveal,1-mover)
    return token


def canonical_observer_fen(fen):
    fields=fen.split()
    if len(fields)!=5 or fields[1] not in ('w','b'):raise ValueError('Invalid observer FEN')
    pool=fields[2];counts={}
    if pool!='-':
        chunks=re.findall(r'([RNBACPrnbacp])([0-9]+)',pool)
        if ''.join(k+v for k,v in chunks)!=pool or len({k for k,_ in chunks})!=len(chunks):
            raise ValueError('Invalid observer FEN inventory')
        counts={k:int(v) for k,v in chunks if int(v)}
    return fields[0],fields[1],tuple(sorted(counts.items())),int(fields[3])


class ObserverHistoryExpert(FixedDepthOpponent):
    """Reuse pinned one-thread UCI transport, with cold fixed-node queries."""
    def position_snapshot(self,tokens,expected_fen):
        if any(re.fullmatch(r'[a-i][0-9][a-i][0-9][RNBACPrnbacp]*',t) is None for t in tokens):
            raise ValueError('Invalid strict observer history token')
        self._send('setoption name Clear Hash')
        self._send('position startpos'+(' moves '+' '.join(tokens) if tokens else ''))
        self._send('d');self._send('moves');self._send('isready')
        lines=self._until('readyok')
        if any('bad move ' in line or 'bad fen' in line for line in lines):
            raise ValueError('Native expert rejected complete source-rule history')
        fens=[line[4:] for line in lines if line.startswith('fen ')]
        legal=[line.split()[1:] for line in lines if line=='moves' or line.startswith('moves ')]
        if len(fens)!=1 or len(legal)!=1:raise ValueError('Native observer echo missing')
        if canonical_observer_fen(fens[0])!=canonical_observer_fen(expected_fen):
            raise ValueError('Native complete-history view differs from actor observation')
        return {'echo_fen':fens[0],'legal_moves':legal[0]}

    def analyse(self,tokens,expected_fen,nodes):
        if type(nodes) is not int or nodes<1:raise ValueError('Require positive fixed-node budget')
        snapshot=self.position_snapshot(tokens,expected_fen)
        self._send('go nodes '+str(nodes));lines=self._until('bestmove ')
        choice=lines[-1].split()[1]
        if choice not in snapshot['legal_moves']:raise ValueError('Expert returned no legal move for source root')
        scores=[]
        for line in lines:
            match=re.search(r'^info depth (\d+).*score (cp|mate) (-?\d+)',line)
            if match and 'lowerbound' not in line and 'upperbound' not in line:
                scores.append({'depth':int(match[1]),'kind':match[2],'value':int(match[3])})
        return {**snapshot,'bestmove':choice,'last_completed_score':scores[-1] if scores else None,
                'requested_nodes':nodes}
