import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)
