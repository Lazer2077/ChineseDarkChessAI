"""Descriptive teacher-search proxies and exact observer-input alias keys."""
import math
import numpy as np


def search_score_proxy(score,scale=860.):
    """Declared monotone search-score proxy, never a calibrated game value."""
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid fixed score scale')
    if not isinstance(score,dict) or type(score.get('value')) is not int or type(score.get('depth')) is not int or score['depth']<1:
        raise ValueError('Require a typed completed search score')
    value=score['value']
    if score.get('kind')=='cp':return math.tanh(value/scale)
    if score.get('kind')=='mate' and value!=0:return math.copysign(1.,value)
    raise ValueError('Unknown or ambiguous mate-zero score')


def observer_input_keys(raw):
    """Exactly the byte fields used by the current encoder, with its D=0 gate.

    Excludes score/result/ply/action labels. This describes exact aliases in
    this representation, not sufficiency of that representation for control.
    """
    if raw.dtype!=np.uint8 or raw.ndim!=2 or raw.shape[1]!=112:raise ValueError('Require byte observer records')
    fields=np.concatenate([raw[:,:103],raw[:,111:112]],axis=1).copy()
    full=((raw[:,:90]&15)!=8).all(1)
    fields[full,91:103]=0
    return [r.tobytes() for r in fields]
