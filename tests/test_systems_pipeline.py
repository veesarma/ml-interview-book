import math

from mlbook.systems import pipeline_calc as pc


def test_bubble_fraction_closed_form():
    assert pc.bubble_fraction(1, 8) == 0.0
    assert math.isclose(pc.bubble_fraction(4, 8), 3 / 11)
    assert math.isclose(pc.bubble_fraction(4, 8, v=2), 3 / 19)
    assert math.isclose(pc.bubble_over_ideal(4, 8), 3 / 8)


def test_simulate_schedule_matches_closed_form_for_gpipe_and_1f1b():
    p, m = 4, 8
    for sch in ("gpipe", "1f1b"):
        ops = pc.simulate_schedule(sch, p, m, t_f=1.0, t_b=2.0)
        assert len(ops) == 2 * p * m
        assert math.isclose(max(o.end for o in ops), (m + p - 1) * 3.0)
        assert math.isclose(pc.simulated_bubble_fraction(ops, p), pc.bubble_fraction(p, m))


def test_peak_in_flight_gpipe_is_m_and_1f1b_is_p_minus_stage():
    p, m = 4, 8
    assert pc.peak_in_flight(pc.simulate_schedule("gpipe", p, m), p) == [m] * p
    assert pc.peak_in_flight(pc.simulate_schedule("1f1b", p, m), p) == [4, 3, 2, 1]
