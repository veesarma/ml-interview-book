import torch

torch.set_num_threads(1)  # the sandbox oversubscribes cores; 1 thread is fastest for these tiny nets
import torch.nn.functional as F

from mlbook.vision.resnet_block import BasicBlock, Bottleneck, MBConv, PreActBlock, SEBlock, TinyResNet
from mlbook.vision.tiny_cnn import TinyCNN, synthetic_shapes


def _train(model, x, y, steps=60, lr=0.05):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def test_tiny_cnn_shapes_and_learns():
    x, y = synthetic_shapes(96)
    model = TinyCNN(1, 4, width=8)
    assert model(x[:2]).shape == (2, 4)
    assert _train(model, x, y) > 0.9


def test_tiny_resnet_learns():
    x, y = synthetic_shapes(96)
    assert _train(TinyResNet(1, 4, width=8), x, y) > 0.9


def test_block_shapes_and_identity_init():
    x = torch.randn(2, 8, 12, 12)
    assert BasicBlock(8, 8)(x).shape == (2, 8, 12, 12)
    assert BasicBlock(8, 16, stride=2)(x).shape == (2, 16, 6, 6)
    assert PreActBlock(8, 8)(x).shape == (2, 8, 12, 12)
    assert Bottleneck(8, 4, stride=2)(x).shape == (2, 16, 6, 6)
    assert MBConv(8, 8, k=3, stride=1)(x).shape == (2, 8, 12, 12)
    assert MBConv(8, 12, k=5, stride=2)(x).shape == (2, 12, 6, 6)
    assert SEBlock(8)(x).shape == x.shape
    # zero-γ init: the residual branch contributes nothing, so y = ReLU(x)
    blk = BasicBlock(8, 8).eval()
    assert torch.allclose(blk(x), torch.relu(x))


def test_residual_gradient_has_identity_path():
    # With the residual branch zeroed, ∂y/∂x is exactly the identity mask of ReLU
    blk = PreActBlock(4, 4).eval()
    with torch.no_grad():
        blk.conv2.weight.zero_()
    x = torch.randn(1, 4, 5, 5, requires_grad=True)
    blk(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
