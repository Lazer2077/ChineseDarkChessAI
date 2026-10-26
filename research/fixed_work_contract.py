"""Public-only fixed-work witness contract; independent of either rules module."""
import hashlib
import math
import re

INITIAL_BOARD = 'XXXXKXXXX'+'.'*10+'X'+'.'*5+'X'+'.'+'X.X.X.X.X'+'.'*18+'x.x.x.x.x'+'.'+'x'+'.'*5+'x'+'.'*10+'xxxxkxxxx'


def public_key(board, actor, history):
    return hashlib.sha256((board+'|'+actor+'|'+' '.join(history)).encode()).hexdigest()


def validate_record(record, simulations, strategy):
    if strategy not in ('puct', 'halving', 'public-halving'): raise ValueError('Unknown fixed-work strategy')
    board = record['public_board']
    if len(board) != 90 or set(board)-set('.KRNBACPXkrnbacpx') or record['actor'] not in ('r', 'b'):
        raise ValueError('Invalid public observation')
    full = not ('X' in board or 'x' in board)
    method = 'halving' if strategy == 'halving' or (strategy == 'public-halving' and full) else 'puct'
    if record['strategy'] != strategy or record['actual_method'] != method or record['fully_public'] != full:
        raise ValueError('Incorrect public phase gate')
    if record['requested_simulations'] != simulations or record['completed_simulations'] != simulations:
        raise ValueError('Incorrect completed simulation budget')
    edges = record['root_edges']
    if not edges or record['selected_move'] not in edges: raise ValueError('Missing selected action')
    for move, edge in edges.items():
        n, value = edge['visits'], edge['next_player_value_sum']
        if re.fullmatch('[a-i][0-9][a-i][0-9]', move) is None or type(n) is not int or n < 0:
            raise ValueError('Invalid root visits')
        if not math.isfinite(value) or abs(value) > n+1e-6: raise ValueError('Unbacked virtual credits')
    if sum(edge['visits'] for edge in edges.values()) != simulations or edges[record['selected_move']]['visits'] < 1:
        raise ValueError('Root edges disagree with completed work')
    if not math.isfinite(record['visit_mass']) or abs(record['visit_mass']-1.) > 1e-7:
        raise ValueError('Invalid policy mass')
    leaves = record['nn_leaf_evals_excluding_root']
    if type(leaves) is not int or not 0 <= leaves <= simulations: raise ValueError('Invalid leaf evaluation count')
    return method
