from multiprocessing import Pool
import numpy as np


from mmdet.core.mask.structures import PolygonMasks
import pycocotools.mask as mask_util
from typing import List, Dict
from .mean_ap import average_precision, print_map_summary


def tpfpmiou_func(
        det_masks: List[Dict],
        gt_masks: List[Dict],
        cls_scores,
        iou_thr=0.5):
    # EUGENE: DOCSTRING
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_masks: Detected masks of this image.
        gt_masks: GT bboxes of this image, of shape (n, 4).
        cls_scores: (n, 1)
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    num_dets = len(det_masks)
    num_gts = len(gt_masks)

    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)
    gt_covered_iou = np.zeros(num_gts, dtype=np.float32)

    if len(gt_masks) == 0:
        fp[...] = 1
        return tp, fp, np.mean(gt_covered_iou)
    if num_dets == 0:
        return tp, fp, np.mean(gt_covered_iou)

    ious = mask_util.iou(det_masks, gt_masks, len(gt_masks) * [0])
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-cls_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)
    # if no area range is specified, gt_area_ignore is all False
    for i in sort_inds:
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                gt_covered_iou[matched_gt] = ious_max[i]
                tp[i] = 1
            else:
                fp[i] = 1
            # otherwise ignore this detected bbox, tp = 0, fp = 0
        else:
            fp[i] = 1
    return tp, fp, np.mean(gt_covered_iou)


def get_cls_results(det_results, annotations, class_id):
    # EUGENE: DOCSTRING
    cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]

    # cls_dets = [img_res[1][class_id] for img_res in det_results]
    cls_dets = []
    for i, det in enumerate(det_results):
        det_masks = det[1][class_id]
        cls_dets.append([])
        for det_mask in det_masks:
            if isinstance(det_mask, np.ndarray):
                cls_dets[i].append(
                  mask_util.encode(
                    np.array(
                      det_mask[:, :, np.newaxis], order='F', dtype='uint8'))[0])
            else:
              cls_dets[i].append(det_mask)

    cls_gts = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        if isinstance(ann['masks'], PolygonMasks):
            masks = ann['masks'].to_ndarray()[gt_inds]
            encoded_masks = [
                mask_util.encode(
                    np.array(m[:, :, np.newaxis], order='F', dtype='uint8')
                )[0] for m in masks]
            cls_gts.append(encoded_masks)
        elif isinstance(ann['masks'], list):
            cls_gts.append([])
        else:
            raise RuntimeError("UNKNOWN ANNOTATION FORMAT")

    return cls_dets, cls_gts, cls_scores


def eval_segm(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
        nproc=4):
    # EUGENE: DOCSTRING
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0][0])

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_results = get_cls_results(det_results, annotations, i)
        cls_dets, cls_gts, cls_scores = cls_results

        # compute tp and fp for each image with multiple processes
        tpfpmiou = pool.starmap(
            tpfpmiou_func,
            zip(cls_dets, cls_gts, cls_scores,
                [iou_thr for _ in range(num_imgs)]))
        tp, fp, miou = tuple(zip(*tpfpmiou))

        # sort all det bboxes by score, also sort tp and fp
        cls_scores = np.hstack(cls_scores)
        num_dets = cls_scores.shape[0]
        num_gts = np.sum([len(cls_gts) for cls_gts in cls_gts])
        sort_inds = np.argsort(-cls_scores)
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap,
            'miou': miou
        })
    pool.close()

    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, None, logger=logger)

    return mean_ap, eval_results
