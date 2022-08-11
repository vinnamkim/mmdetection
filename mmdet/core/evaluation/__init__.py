# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from .class_names import (cityscapes_classes, coco_classes, dataset_aliases,
                          get_classes, imagenet_det_classes,
                          imagenet_vid_classes, voc_classes)
from .eval_hooks import (DistEvalHook, DistEvalPlusBeforeRunHook, EvalHook,
                         EvalPlusBeforeRunHook)
from .mean_ap import average_precision, eval_map, print_map_summary
from .recall import (eval_recalls, plot_iou_recall, plot_num_recall,
                     print_recall_summary)
from .text_evaluation import text_eval
from .mean_ap_seg import eval_segm
from .mae import MAE, CustomMAE


__all__ = [
    'voc_classes', 'imagenet_det_classes', 'imagenet_vid_classes',
    'coco_classes', 'cityscapes_classes', 'dataset_aliases', 'get_classes',
    'DistEvalHook', 'DistEvalPlusBeforeRunHook', 'EvalHook', 'EvalPlusBeforeRunHook',
    'average_precision', 'eval_map', 'print_map_summary', 'eval_recalls',
    'print_recall_summary', 'plot_num_recall', 'plot_iou_recall', 'text_eval',
    'eval_segm', 'MAE', 'CustomMAE'
]
