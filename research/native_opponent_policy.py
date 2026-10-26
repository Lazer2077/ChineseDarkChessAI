"""Count-based opponent-policy approximation using unchanged jqe forced searches.

Only public board/counts plus a hypothesized private inventory enter the model.
Each action receives a cold-hash fixed-depth score; its softmax/epsilon mixture
is a declared policy approximation, not the original engine's timed policy.
"""
from collections import deque
import math
import os
import re
import selectors
import subprocess
import time


def hypothetical_fen(board, actor, public, observer, captured, no_capture):
    if len(board) != 90 or actor not in (0, 1) or observer != 1-actor:
        raise ValueError('Require a public board and opponent decision')
    pools = [list(row) for row in public]
    if len(captured) != 6 or len(pools) != 2 or any(len(row) != 6 for row in pools):
        raise ValueError('Require six-type inventories')
    pools[observer] = [k-h for k, h in zip(pools[observer], captured)]
    if any(not isinstance(n, int) or n < 0 for row in pools for n in row):
        raise ValueError('Impossible inventory')
    ranks = []
    for r in range(9, -1, -1):
        rank = ''; empty = 0
        for pc in board[r*9:r*9+9]:
            if pc == '.': empty += 1; continue
            if pc not in 'KRNBACPXkrnbacpx': raise ValueError('Invalid public piece')
            if empty: rank += str(empty); empty = 0
            rank += pc
        if empty: rank += str(empty)
        ranks.append(rank)
    urn = ''.join((t if c == 0 else t.lower())+str(n)
                  for c, row in enumerate(pools) for t, n in zip('RNBACP', row) if n)
    return '/'.join(ranks)+(' w ' if actor == 0 else ' b ')+(urn or '-')+' '+str(no_capture)+' 1'


def score_policy(scores, temperature=200., epsilon=.1):
    if not scores or not math.isfinite(temperature) or temperature <= 0 or not 0 < epsilon <= 1:
        raise ValueError('Invalid policy parameters')
    if not all(math.isfinite(s) for s in scores.values()): raise ValueError('Nonfinite score')
    maximum = max(scores.values())
    weights = {a: math.exp((s-maximum)/temperature) for a, s in scores.items()}
    z = math.fsum(weights.values())
    return {a: (1-epsilon)*p/z+epsilon/len(scores) for a, p in weights.items()}


class FixedDepthOpponent:
    def __init__(self, binary, network, depth=2, timeout=30.):
        if not isinstance(depth, int) or depth < 1: raise ValueError('Invalid depth')
        self.depth, self.timeout = depth, timeout
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self.selector = selectors.DefaultSelector(); self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        self.buffer = b''; self.lines = deque()
        try:
            self._send('uci'); self._until('uciok')
            for option in ['Threads value 1', 'Hash value 16', 'NNUEQuant value true', 'EvalFile value '+str(network)]:
                self._send('setoption name '+option)
            self._send('isready'); self._until('readyok')
        except BaseException: self.close(); raise

    def _send(self, line):
        self.proc.stdin.write((line+'\n').encode())

    def _until(self, prefix):
        result = []; deadline = time.monotonic()+self.timeout
        while True:
            if self.lines:
                line = self.lines.popleft()
                if 'bad fen' in line or 'load failed' in line: raise ValueError(line.split(':', 1)[0])
                result.append(line)
                if line.startswith(prefix): return result
                continue
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining): raise TimeoutError('Own opponent-model process timed out')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk: raise RuntimeError('Own opponent-model process closed')
            self.buffer += chunk
            parts = self.buffer.split(b'\n'); self.buffer = parts.pop()
            self.lines.extend(p.decode().strip() for p in parts)

    def scores(self, fen, legal_moves):
        moves = list(legal_moves)
        if not moves or len(set(moves)) != len(moves): raise ValueError('Invalid public mask')
        result = {}
        for move in moves:
            if re.fullmatch('[a-i][0-9][a-i][0-9]', move) is None: raise ValueError('Invalid move')
            self._send('setoption name Clear Hash')
            self._send('position fen '+fen)
            self._send('go depth '+str(self.depth)+' searchmoves '+move)
            lines = self._until('bestmove ')
            if lines[-1].split()[1] != move: raise ValueError('Forced move not respected')
            found = []
            for line in lines:
                match = re.search(r'^info depth (\d+).*score (cp|mate) (-?\d+)', line)
                if match and 'lowerbound' not in line and 'upperbound' not in line:
                    value = int(match.group(3))
                    if match.group(2) == 'mate':
                        # Declared mate-score policy mapping, not an exact
                        # inverse of UCI's rounded full-move mate distance.
                        value = (1 if value > 0 else -1)*(32000-2*abs(value))
                    found.append((int(match.group(1)), value))
            if not found or found[-1][0] != self.depth: raise ValueError('Fixed depth not completed')
            result[move] = found[-1][1]
        return result

    def close(self):
        if hasattr(self, 'proc'):
            if self.proc.poll() is None:
                try: self._send('quit'); self.proc.wait(timeout=2)
                except (BrokenPipeError, subprocess.TimeoutExpired): self.proc.kill(); self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout): stream.close()
        if hasattr(self, 'selector'): self.selector.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
