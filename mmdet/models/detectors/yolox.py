from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    """Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
