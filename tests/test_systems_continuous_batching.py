from mlbook.systems import continuous_batching_sim as cb


def test_both_schedulers_conserve_tokens_and_complete_all_requests():
    w = cb.make_workload(100, rate=50.0)
    for sim in (cb.simulate_static, cb.simulate_continuous):
        m = sim(w, max_batch=8)
        assert len(m.completed) == 100
        assert m.output_tokens == sum(r.gen_len for r in w)


def test_simulate_continuous_beats_simulate_static_on_variable_lengths():
    w = cb.make_workload(200, rate=50.0, mean_gen=64)
    s, c = cb.simulate_static(w, 16), cb.simulate_continuous(w, 16)
    assert c.throughput > 1.5 * s.throughput
    assert c.mean_latency < s.mean_latency


def test_schedulers_agree_when_all_requests_are_identical():
    reqs = [cb.Request(i, 0.0, 64, 32) for i in range(8)]
    s, c = cb.simulate_static(reqs, 8), cb.simulate_continuous(reqs, 8)
    assert abs(s.makespan - c.makespan) < 1e-9
