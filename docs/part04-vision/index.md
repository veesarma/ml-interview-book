# Part IV: Computer vision

Vision is home turf for most senior perception engineers, which is why interviewers push past the engineering layer into the reasons: why a stride-2 convolution aliases, why a residual connection changes optimisation without changing expressivity, why focal loss needs a prior-initialised bias, why a mask head needs ROIAlign, why depth error grows with the square of range. This part runs from pixels to 3-D, deriving every result you might be asked to reproduce at a whiteboard and implementing every algorithm you might be asked to code.

## What is in this part

| Chapter | You will be able to |
|---|---|
| [Image representation & signal processing](01-image-representation.md) | Explain sampling and aliasing through the Fourier lens, implement Gaussian, Sobel and Laplacian filtering, pyramids and bilinear sampling, and say why cameras output Bayer and YUV and why FPN is a Laplacian pyramid in disguise. |
| [Convolutions](02-convolutions.md) | Derive output size, receptive field, FLOPs and parameter counts, implement a conv layer (naive and im2col) with forward and backward matching `torch.nn.functional.conv2d`, and explain transposed conv, checkerboards, depthwise separable and dilated convs. |
| [CNN architectures](03-cnn-architectures.md) | Tell the LeNet to AlexNet to VGG to ResNet to EfficientNet to ConvNeXt story with the reason for every step, derive why $x_{l+1} = x_l + F(x_l)$ helps, and know the ImageNet recipe evolution and the mobile deployment trade-offs. |
| [Object detection](04-detection.md) | Work fluently with anchors, IoU variants, NMS, two-stage against one-stage, focal loss derived with its gradient, FPN, FCOS, the YOLO lineage and DETR's shift to set prediction, and implement IoU, box coding, anchors, assignment, NMS and a single-stage loss from scratch. |
| [Segmentation](05-segmentation.md) | Define semantic, instance and panoptic segmentation and their metrics, explain FCN, U-Net, Mask R-CNN, DeepLab with ASPP, SegFormer, Mask2Former's mask classification and SAM's promptable design, and implement a U-Net with Dice and boundary losses. |
| [Geometry & cameras](06-geometry.md) | Derive $p \sim K[R\,|\,t]P$, intrinsics, extrinsics and distortion, epipolar geometry with the normalised 8-point algorithm, DLT triangulation, PnP, Lucas-Kanade and the sparsity structure of bundle adjustment, then implement all of it and test on synthetic scenes. |
| [3D perception](07-3d-perception.md) | Compare point, voxel and pillar representations, give PointNet's invariance argument, explain sparse convolution, compare LiDAR and radar, derive stereo and monocular depth, and place occupancy as the modern output. |

## Prerequisites

* [Linear algebra](../part01-math/01-linear-algebra.md) for SVD (the 8-point algorithm, DLT), projections and null spaces.
* [Calculus & matrix calculus](../part01-math/02-calculus-matrix-calculus.md) for the chain rule through reshapes (conv backward) and Jacobians (bundle adjustment).
* [Backpropagation](../part03-neural-nets/02-backpropagation.md) and [normalization](../part03-neural-nets/05-normalization.md), since BatchNorm sits inside every block here.
* [Tensor shapes & broadcasting](../part01-math/07-tensor-shapes-broadcasting.md), because you will read `(B, C, H, W)` on every line.

## If you have one day

1. Morning, 3 hours. [Convolutions](02-convolutions.md) sections 1 to 3: derive output size and receptive field, then code im2col forward and backward. Then [CNN architectures](03-cnn-architectures.md) section 2 for the ResNet derivation and section 5 for the recipes.
2. Midday, 2 hours. [Detection](04-detection.md): the interview card, section 2 for IoU and focal loss with its gradient, and section 3 for the IoU, NMS and assignment code. This is the most-asked coding round for perception roles.
3. Afternoon, 2 hours. [Geometry](06-geometry.md) sections 1 and 2 for projection and the epipolar constraint, and the [3D perception](07-3d-perception.md) interview card for disparity to depth and PointNet invariance.
4. Evening, 1 hour. The interview cards of [Image representation](01-image-representation.md) and [Segmentation](05-segmentation.md), then skim the production case studies in every chapter so you can name who used what.

Every chapter carries a "Retype by hand" section listing exactly which functions to reproduce from memory, which to read, the pytest command that checks them, and a target time. Working through those seven lists is the coding-round preparation for this part.

## How this part connects onward

* Vision Transformers, DETR's Hungarian matching and CLIP are in [Part VIII](../part08-multimodal/index.md). This part covers the convolutional baseline they were measured against and the detection vocabulary they inherit.
* Multi-camera BEV, sensor fusion, tracking and occupancy, which together form the autonomous-vehicle stack, are in [Part XI](../part11-perception-autonomy/index.md). The camera model and the 3-D representations here are its foundation.
* Detection and segmentation metrics (mAP, mIoU, PQ) are defined here and treated as evaluation methodology in [Part XIII](../part13-retrieval-eval-reliability/02-evaluation.md).
* The [AV perception system design](../part17-ml-system-design/05-perception-system-av.md) and [OCR system design](../part17-ml-system-design/09-ocr-document-understanding.md) chapters assume everything in this part.

All code is in `src/mlbook/vision`, `src/mlbook/detection` and `src/mlbook/geometry`, tested by `pytest tests/test_vision_* tests/test_detection_* tests/test_geometry_*`.
