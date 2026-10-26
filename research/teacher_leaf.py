"""Attributed teacher static value, evaluated only from actor observations."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
from research.neural_search import ObservationPUCT
from research.policy_value import AZ

ROOT = Path(__file__).resolve().parents[1]
_LOADED = None


def observer_fen(info):
    board = []
    # Yuan index0 is a9; native jqe FEN also starts at rank9.
    for r in range(10):
        rank = ''; empty = 0
        for pc in info.board[r*9:r*9+9]:
            if not pc:
                empty += 1; continue
            if empty: rank += str(empty); empty = 0
            letter = 'X' if AZ.jq.is_dark(pc) else 'KRNBACP'[AZ.jq.piece_type(pc)-1]
            rank += letter if AZ.jq.piece_color(pc) == AZ.jq.RED else letter.lower()
        if empty: rank += str(empty)
        board.append(rank)
    letters = dict(zip(AZ.jq.POOL_TYPES, 'RNBACP'))
    pool = ''.join((letters[t] if c == AZ.jq.RED else letters[t].lower())+str(info.pool_known[c][t])
                   for c in (AZ.jq.RED, AZ.jq.BLACK) for t in AZ.jq.POOL_TYPES if info.pool_known[c][t])
    return '/'.join(board)+(' w ' if info.side == AZ.jq.RED else ' b ')+(pool or '-')+' '+str(info.no_capture_plies)+' 1'


class TeacherStaticValue:
    def __init__(self, network, library=ROOT/'build/nnue-observer/libobserver_nnue.so', scale=860.):
        global _LOADED
        network, library = Path(network), Path(library)
        if not math.isfinite(scale) or scale <= 0: raise ValueError('Invalid value scale')
        manifest = json.loads(library.with_name('manifest.json').read_text())
        if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['library_sha256']:
            raise ValueError('NNUE library changed')
        for p, digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest: raise ValueError('NNUE source changed')
        identity = (str(library.resolve()), hashlib.sha256(network.read_bytes()).hexdigest())
        if _LOADED is not None and _LOADED != identity:
            raise ValueError('Use an isolated process for different native NNUE models')
        self.lib = ctypes.CDLL(str(library.resolve())); self.scale = scale
        self.lib.cdca_load_nnue.argtypes = [ctypes.c_char_p]; self.lib.cdca_load_nnue.restype = ctypes.c_char_p
        self.lib.cdca_eval_observer_fen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.cdca_eval_observer_fen.restype = ctypes.c_char_p
        self._check(self.lib.cdca_load_nnue(str(network.resolve()).encode()))
        _LOADED = identity

    @staticmethod
    def _check(error):
        if error is not None: raise ValueError(error.decode())

    def centipawns(self, info):
        result = ctypes.c_int()
        self._check(self.lib.cdca_eval_observer_fen(observer_fen(info).encode(), ctypes.byref(result)))
        return result.value

    def __call__(self, info):
        return math.tanh(self.centipawns(info)/self.scale)


class TeacherLeafPUCT(ObservationPUCT):
    def __init__(self, net, device='cpu', seed=20261029, leaf_batch=8, teacher=None):
        super().__init__(net, device, seed, leaf_batch)
        self.teacher = teacher or TeacherStaticValue(ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn')

    def _evaluate_batch(self, infos, histories, *, repetition_entries=None):
        policy, _ = super()._evaluate_batch(infos, histories, repetition_entries=repetition_entries)
        return policy, [self.teacher(info) for info in infos]
