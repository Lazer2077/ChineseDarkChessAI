from scripts.screen_baselines import summary


def game(red,result=1.,pair=0,reason='BLACK_LOSES'):
    return {'red':red,'result':result,'pair':pair,'reason':reason}


def test_color_swaps_are_whole_statistical_units_and_not_promotion():
    report=summary([game('candidate'),game('baseline',0.)],[],2)
    assert report['wins']==2 and report['complete_pairs']==1
    assert report['valid_technical_run']
    assert report['paired_bootstrap_95']==[1.,1.]
    assert report['paired_hoeffding_95'][0]<.5
    assert not report['single_match_superiority_gate'] and not report['promotion_supported']


def test_duplicate_color_or_voids_do_not_count_as_clean_strength():
    duplicate=summary([game('candidate'),game('candidate')],[],2)
    assert duplicate['complete_pairs']==0 and not duplicate['valid_technical_run']
    void=summary([game('candidate'),game('baseline',0.)],[{'reason':'failure'}],2)
    assert not void['valid_technical_run']


def test_technical_wins_are_reported_but_not_eligible():
    report=summary([game('candidate',reason='TIME_FORFEIT baseline'),game('baseline',0.)],[],2)
    assert report['wins']==2 and report['technical_results']
    assert not report['valid_technical_run']
