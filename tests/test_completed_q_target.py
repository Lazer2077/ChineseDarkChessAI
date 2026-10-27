import numpy as np
import pytest
from research.completed_q_target import completed_q_target


def test_mixed_completion_uses_prior_weights_not_visit_weighted_average():
    r=completed_q_target([.4,.4,.2],[1,99,0],[1.,-1.,np.nan],.5,2.)
    assert r['mixed_value']==pytest.approx(.5/101)
    assert r['completed_q'][2]==pytest.approx(.5/101)
    assert r['policy'].sum()==pytest.approx(1)
    assert r['estimated_completed_q_gain']>0


def test_zero_scale_or_no_visits_preserves_prior_and_support():
    prior=np.asarray([.2,.8,0.])
    assert np.array_equal(completed_q_target(prior,[0,0,0],[np.nan]*3,.1,100)['policy'],prior)
    assert np.array_equal(completed_q_target(prior,[1,2,0],[1.,-1.,np.nan],0.,0)['policy'],prior)
    assert completed_q_target(prior,[1,2,0],[1.,-1.,np.nan],0.,20)['policy'][2]==0
    with pytest.raises(ValueError,match='Nonfinite visited'):
        completed_q_target(prior,[1,0,0],[np.nan]*3,0.,1.)


def test_completed_vector_proxy_gain_over_many_finite_examples():
    rng=np.random.default_rng(1004)
    for _ in range(100):
        prior=rng.dirichlet(np.ones(7));visits=rng.integers(0,9,size=7)
        q=rng.uniform(-1,1,7);r=completed_q_target(prior,visits,q,0.,rng.uniform(0,100))
        assert r['estimated_completed_q_gain']>=-1e-10
        assert r['target_to_prior_kl']>=-1e-10
