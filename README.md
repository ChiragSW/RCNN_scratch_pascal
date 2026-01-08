# Faster R-CNN Training Report

**Context & Key Numbers**
- Model: Faster R-CNN with a ResNet50-FPN backbone (backbone used to increase acc).
- Training setup highlights: backbone frozen for the first epoch, then unfrozen; AdamW optimizer; ReduceLROnPlateau scheduler; mixed-precision (GradScaler); early stopping with patience.
- Reported metrics from the run: average training loss ≈ 0.3225, validation loss ≈ 0.3211, and mAP@0.5 ≈ 0.7698.
- Saved model: `faster_rcnn_scratch.h5` (weights exported via `h5py`).

**1. Architecture design choices**

- Why Faster R-CNN + ResNet50-FPN?
  - Faster R-CNN is a two-stage detector: a Region Proposal Network (RPN) proposes candidate regions, and a second-stage head classifies and refines boxes. This design tends to give stronger localization and higher accuracy than single-stage detectors for medium-sized datasets like Pascal VOC.
  - ResNet50 with an FPN (feature pyramid) offers a good trade-off: modern, well-regularized residual blocks for representation power, and FPN provides strong multi-scale feature maps which help detect objects of different sizes.

- Modifications used here
  - Classifier head adjusted to the dataset classes (VOC-type labels). The number of output units in the classifier and bbox heads was changed to match the label set.
  - Backbone frozen for the initial epoch: reduces noisy gradients early and stabilizes the pre-trained feature extractor while the detection heads adapt.


**2. Data augmentation strategies**

- Random horizontal flip (50%): a simple, safe, and common augmentation that doubles left-right appearance variability.
- Random scaling / multi-scale resizing: randomly resize the shorter side (e.g., 480–800 px). Helps robustness across object sizes.
- Random crop + random paste / random translate: simulates partial objects and different framing.
- Color jitter (brightness, contrast, saturation, hue): helps robustness to lighting and sensor differences.
- Gaussian blur or small JPEG compression noise: sometimes improves generalization across imaging conditions.
- Photometric distortions: complementary to color jitter.

**3. Training methodology**

- Optimizer: AdamW with lr=1e-4 and weight_decay=1e-4.
- Mixed-precision training via `GradScaler()` and `autocast()` for faster training + reduced memory usage.
- Scheduler: ReduceLROnPlateau (monitors validation loss; factor=0.5, patience as configured) to reduce learning rate when performance plateaus.
- Early stopping: stops training when validation loss fails to improve for `PATIENCE` epochs.
- Batch sizes: training batch size = 4, validation batch size = 1 (common for detection because images differ in size).
- Unfreeze schedule: backbone frozen for epoch 1 and unfrozen from epoch 2.

**4. Results comparison and interpretation**

What the run shows
- Validation mAP@0.5 = ~0.77
- Training and validation losses are close (0.3225 & 0.3211)

**5. Trade-offs between accuracy and speed**

- Accuracy improvements often require larger backbones, more training time, or stronger augmentations — all increase compute and inference latency.
- If deployment latency is critical (real-time), consider:
  - Using a lighter backbone (MobileNetV2, EfficientNet-lite), or a single-stage detector (YOLO, SSD, RetinaNet).
  - Reducing input resolution at inference time (reduces accuracy for very small objects).
  - Post-training quantization (INT8) and pruning — these can reduce latency with modest accuracy loss if applied carefully.
