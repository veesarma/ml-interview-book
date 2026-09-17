import torch

from mlbook.interp import gradcam as gc


def test_grad_cam_shape_range_and_localisation():
    torch.manual_seed(0)
    model = gc.TinyCNN(n_classes=2, width=4)
    # Hand-set a detector: conv2 is identity-like on channel 0, fc reads channel 0 -> class 1.
    with torch.no_grad():
        model.conv1.weight.zero_(); model.conv1.bias.zero_()
        model.conv1.weight[0, 0, 1, 1] = 1.0  # channel 0 copies the input
        model.conv2.weight.zero_(); model.conv2.bias.zero_()
        model.conv2.weight[0, 0, 1, 1] = 1.0
        model.fc.weight.zero_(); model.fc.bias.zero_()
        model.fc.weight[1, 0] = 1.0
    x = torch.zeros(1, 8, 8)
    x[0, 1:3, 5:7] = 1.0  # bright blob top-right
    cam = gc.grad_cam(model, x, target=1)
    assert cam.shape == (8, 8) and float(cam.min()) >= 0 and abs(float(cam.max()) - 1.0) < 1e-6
    assert cam[1:3, 5:7].mean() > 0.9 and cam[5:8, 0:3].mean() < 0.1
