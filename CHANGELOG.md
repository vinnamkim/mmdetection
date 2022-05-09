# Changelog

All notable changes to this project will be documented in this file.

This project is based on [MMDetection project by OpenMMLab](https://github.com/open-mmlab/mmdetection).

With respect to it we made the following changes.

## \[2022-05-09\]
### Added
* Object counting models using Mask-RCNN
* Rotated object detection models using Mask-RCNN
* YoloX with NNCF support
* Stop training on NaN losses
* FP16 support

### Removed
* Tasks & model templates (moved to OTE)


## \[2021-12-27\]
### Added
* OTE SDK Task interface implementation
* OpenVINO export including alternative export for SSD models
* Models, datasets, metrics for text detection and recognition
* Backbones from pytorchcv and EfficientDet architecture
* Model compression possibility by NNCF and config examples
* Configuration files for
  * custom object detection
  * face detection
  * horizontal text detection
  * person detection
  * person-vehicle-bike detection
  * vehicle detection
* Extra augmentstions and transforms, ReduceOnPlateau LR scheduler, early stopping hook
* Extra tool scripts for
  * COCO dataset visualization and filtration
  * clustering anchor boxes for SSD anchor generator
  * validation of the ONNX and OpenVINO models

### Changed
* Improved the export to the ONNX including runtime mode
* Added OpenVINO installation to the dockerfile configuration
