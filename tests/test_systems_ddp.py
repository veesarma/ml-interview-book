import pytest
import torch

from mlbook.systems import ddp_example as ddp


def test_full_batch_gradients_deterministic():
    a, b = ddp.full_batch_gradients(), ddp.full_batch_gradients()
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_run_ddp_demo_two_processes_gloo():
    try:
        out = ddp.run_ddp_demo(world_size=2)
    except Exception as e:  # spawn / gloo unavailable in this sandbox
        pytest.skip(f"torch.distributed spawn unavailable: {e!r}")
    assert out["manual_vs_full"] < 1e-5
    assert out["ddp_vs_full"] < 1e-5
    assert out["rank_agreement"] == 0.0
