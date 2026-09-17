# Part IV: Computer vision

Vision is where most senior perception engineers have their home turf, which is exactly why interviewers push past the engineering layer into the *why*: why a stride-2 convolution aliases, why a residual connection changes optimisation rather than expressivity, why focal loss needs a prior-initialised bias, why a mask head needs ROIAlign and not ROIPool, why depth error grows with the square of range. This part goes from pixels to 3-D, deriving every result you might be asked to reproduce at a whiteboard and implementing every algorithm you might be asked to code.

## What is in this part

| Chapter | You will be able to |
|---|---|
| [Image representation & signal processing](01-image-representation.md) | Explain sampling and aliasing through the Fourier lens; implement Gaussian/Sobel/Laplacian filtering, pyramids and bilinear sampling; say why cameras output Bayer/YUV and why FPN is a Laplacian pyramid in disguise. |
| [Convolutions](02-convolutions.md) | Derive output size, receptive field, FLOPs and parameter counts; implement a conv layer (naive + im2col) with forward *and* backward matching `torch.nn.functional.conv2d`; explain transposed conv, checkerboards, depthwise separable and dilated convs. |
| [CNN architectures](03-cnn-architectures.md) | Tell the LeNet → AlexNet → VGG → ResNet → EfficientNet → ConvNeXt story with the *reason* for every step; derive why $x_{l+1} = x_l + F(x_l)$ helps; know the ImageNet recipe evolution and the mobile deployment trade-offs. |
| [Object detection](04-detection.md) | Master anchors, IoU variants, NMS, two-stage vs one-stage, focal loss (derived with gradient), FPN, FCOS, YOLO lineage, DETR's shift to set prediction; implement IoU, box coding, anchors, assignment, NMS and a single-stage loss from scratch. |
| [Segmentation](05-segmentation.md) | Define semantic/instance/panoptic and their metrics; explain FCN, U-Net, Mask R-CNN, DeepLab/ASPP, SegFormer, Mask2Former's mask classification and SAM's promptable design; implement a U-Net, Dice and boundary losses. |
| [Geometry & cameras](06-geometry.md) | Derive $p \sim K[R\,|\,t]P$, intrinsics/extrinsics/distortion, epipolar geometry (normalised 8-point), DLT triangulation, PnP, Lucas–Kanade and the structure of bundle adjustment; implement all of it and test on synthetic scenes. |
| [3D perception](07-3d-perception.md) | Compare point/voxel/pillar representations, PointNet's invariance argument, sparse convolution, LiDAR vs radar, stereo and monocular depth (disparity ↔ depth, photometric self-supervision), and occupancy as the modern output. |

## Prerequisites

* [Linear algebra](../part01-math/01-linear-algebra.md). SVD (8-point, DLT), projections, null spaces.
* [Calculus & matrix calculus](../part01-math/02-calculus-matrix-calculus.md). chain rule through reshapes (conv backward), Jacobians (bundle adjustment).
* [Backpropagation](../part03-neural-nets/02-backpropagation.md) and [normalization](../part03-neural-nets/05-normalization.md). BatchNorm is inside every block here.
* [Tensor shapes & broadcasting](../part01-math/07-tensor-shapes-broadcasting.md). you will read `(B, C, H, W)` on every line.

## If you have one day

1. **Morning (3 h):** [Convolutions](02-convolutions.md) §1–3 (derive output size and receptive field, code im2col forward/backward), then [CNN architectures](03-cnn-architectures.md) §2 (the ResNet derivation) and §5 (recipes).
2. **Midday (2 h):** [Detection](04-detection.md) TL;DR, §2 (IoU, focal loss with gradient), §3 (IoU + NMS + assignment code). This is the most-asked coding round for perception roles.
3. **Afternoon (2 h):** [Geometry](06-geometry.md) §1–2 (projection, epipolar constraint) and the [3D perception](07-3d-perception.md) TL;DR (disparity ↔ depth, PointNet invariance).
4. **Evening (1 h):** the interview cards of [Image representation](01-image-representation.md) and [Segmentation](05-segmentation.md); skim the production case studies in every chapter so you can name who used what.

## How this part connects onward

* Vision Transformers, DETR's Hungarian matching and CLIP live in [Part VIII](../part08-multimodal/index.md); this part explains the *convolutional* baseline they were measured against and the detection vocabulary they inherit.
* Multi-camera BEV, sensor fusion, tracking and occupancy. the autonomous-vehicle stack. live in [Part XI](../part11-perception-autonomy/index.md); the camera model and 3-D representations here are their foundation.
* Detection and segmentation metrics (mAP, mIoU, PQ) are defined here and treated as evaluation methodology in [Part XIII](../part13-retrieval-eval-reliability/02-evaluation.md).
* The [AV perception system design](../part17-ml-system-design/05-perception-system-av.md) and [OCR system design](../part17-ml-system-design/09-ocr-document-understanding.md) chapters assume everything in this part.

All code is in `src/mlbook/vision`, `src/mlbook/detection` and `src/mlbook/geometry`, tested by `pytest tests/test_vision_* tests/test_detection_* tests/test_geometry_*`.
