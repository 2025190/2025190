## Framework Overview

### `framework_twostage`
This folder is designed for two-stage training:
1. Install the official ultralytics package
2. Run `train_yolo.py` for initial training
3. After preparing all intermediate prediction results, run `train_stage_two.py` for the second stage of training

### `framework_tri`
This folder is designed for a single framework implementation:
- Contains numerous modified ultralytics libraries
- Includes mAP calculation metrics based on YOLO's computation method
- Installation: Install our custom ultralytics package
- Training: Directly run `train.py` to start training
