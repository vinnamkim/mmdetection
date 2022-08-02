# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import os.path as osp
import tempfile
import uuid
from time import time
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.ops import nms
from tqdm import tqdm

from mmdet.core import BitmapMasks, bbox2result


def timeit(func):

    def wrapper(*args, **kwargs):
        begin = time()
        result = func(*args, **kwargs)
        print(f'\n==== {func.__name__}: {time() - begin} sec ====\n')
        return result

    return wrapper


class Tile:
    """Tile and merge datasets.

    Args:
        dataset (CustomDataset): the dataset to be tiled.
        tile_size (int): the length of side of each tile
        overlap (float, optional): ratio of each tile to overlap with each of
            the tiles in its 4-neighborhood. Defaults to 0.3.
        min_area_ratio (float, optional): the minimum overlap area ratio
            between a tiled image and its annotations. Ground-truth box is
            discarded if the overlap area is less than this value.
            Defaults to 0.2.
        iou_threshold (float, optional): IoU threshold to be used to suppress
            boxes in tiles' overlap areas. Defaults to 0.45.
        max_per_img (int, optional): if there are more than max_per_img bboxes
            after NMS, only top max_per_img will be kept. Defaults to 200.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Defaults to True.
    """

    def __init__(self,
                 dataset,
                 tmp_dir: tempfile.TemporaryDirectory,
                 tile_size: int,
                 overlap: float = 0.3,
                 min_area_ratio: float = 0.2,
                 iou_threshold: float = 0.45,
                 max_per_img: int = 200,
                 filter_empty_gt: bool = True):

        self.min_area_ratio = min_area_ratio
        self.filter_empty_gt = filter_empty_gt
        self.iou_threshold = iou_threshold
        self.max_per_img = max_per_img
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.num_images = len(dataset)
        self.num_classes = len(dataset.CLASSES)
        self.CLASSES = dataset.CLASSES
        self.tmp_folder = tmp_dir.name

        self.__dataset = dataset
        self.__tiles = self.__gen_tile_ann()
        self.__cache_tiles()

    @timeit
    def __cache_tiles(self):
        pbar = tqdm(total=len(self.__tiles))
        pre_img_idx = None
        for i, tile in enumerate(self.__tiles):
            tile['tile_path'] = osp.join(
                self.tmp_folder, "_".join([str(i), tile['uuid'], tile['ori_filename'], '.jpg']))
            x_1, y_1, x_2, y_2 = tile['tile_box']
            dataset_idx = tile['dataset_idx']
            if dataset_idx != pre_img_idx:
                ori_img = self.__dataset[dataset_idx]['img']
                pre_img_idx = dataset_idx

            mmcv.imwrite(ori_img[y_1:y_2, x_1:x_2, :], tile['tile_path'])
            pbar.update(1)

    @timeit
    def __gen_tile_ann(self) -> List[Dict]:
        """Generate tile information and tile annotation from dataset.

        Returns:
            List[Dict]: A list of tiles generated from the dataset. Each item
              comprises tile annotation and tile coordinates relative to the
              original image.
        """
        tiles = []
        pbar = tqdm(total=len(self.__dataset))
        for idx, result in enumerate(self.__dataset):
            tiles.extend(self.__gen_tiles_single_img(result, dataset_idx=idx))
            pbar.update(1)
        return tiles

    def __gen_tiles_single_img(self, result: Dict, dataset_idx: int) -> List[Dict]:
        """Generate tile annotation for a single image.

        Args:
            result (Dict): the original image-level result (i.e. the original
            image annotation) dataset_idx (int): the image index this tile
            belongs to

        Returns:
            List[Dict]: tile annotation with some other useful information for
              data pipeline.
        """
        tile_list = []
        gt_bboxes = result.pop('gt_bboxes', np.zeros((0, 4), dtype=np.float32))
        gt_masks = result.pop('gt_masks', None)
        gt_bboxes_ignore = result.pop('gt_bboxes_ignore', np.zeros((0, 4), dtype=np.float32))
        gt_labels = result.pop('gt_labels', np.array([], dtype=np.int64))
        img_shape = result.pop('img_shape')
        height, width = img_shape[:2]
        y_segments = self.__slice_2d(height)
        x_segments = self.__slice_2d(width)
        _tile = self.__prepare_result(result)

        for x_seg in x_segments:
            for y_seg in y_segments:
                x_1, x_2 = x_seg
                y_1, y_2 = y_seg
                tile = copy.deepcopy(_tile)
                tile['original_shape_'] = img_shape
                tile['ori_shape'] = (y_2 - y_1, x_2 - x_1, 3)
                tile['img_shape'] = tile['ori_shape']
                tile['tile_box'] = (x_1, y_1, x_2, y_2)
                tile['dataset_idx'] = dataset_idx
                tile['gt_bboxes_ignore'] = gt_bboxes_ignore
                tile['uuid'] = str(uuid.uuid4())
                self.__tile_ann_assignment(
                    tile,
                    np.array([[x_1, y_1, x_2, y_2]]),
                    gt_bboxes, gt_masks, gt_labels)
                # filter empty ground truth
                if self.filter_empty_gt and len(tile['gt_labels']) == 0:
                    continue
                tile_list.append(tile)
        return tile_list

    def __prepare_result(self, result: Dict) -> Dict:
        """Prepare results dict for pipeline.

        Args:
            result (Dict): original image-level result for a tile

        Returns:
            Dict: result template with useful information for data pipeline.
        """
        result_template = dict(
            ori_filename=result['ori_filename'],
            filename=result['filename'],
            bbox_fields=result['bbox_fields'],
            mask_fields=result['mask_fields'],
            seg_fields=result['seg_fields'],
            img_fields=result['img_fields'],
            # proposal_file=result['proposal_file'],
            # img_info=result['img_info'],
            # img_prefix=result['img_prefix'],
            # seg_prefix=result['seg_prefix'],
        )
        return result_template

    def __tile_ann_assignment(self, tile_result: Dict, tile_box: np.ndarray, gt_bboxes: np.ndarray,
                              gt_masks: BitmapMasks, gt_labels: np.ndarray) -> Dict:
        """Assign new annotation to this tile.

        Ground-truth is discarded if the overlap with this tile is lower than
        min_area_ratio.

        Args:
            tile_box (np.ndarray): the coordinate for this tile box
              (i.e. the tile coordinate relative to the image)
            gt_bboxes (np.ndarray): the original image-level boxes
            gt_labels (np.ndarray): the original image-level labels

        Returns:
            Dict: bboxes, masks in this tile, labels in this tile
        """
        x_1, y_1 = tile_box[0][:2]
        overlap_ratio = self.__tile_boxes_overlap(tile_box, gt_bboxes)
        match_idx = np.where((overlap_ratio[0] >= self.min_area_ratio))[0]

        if len(match_idx):
            tile_lables = gt_labels[match_idx][:]
            tile_bboxes = gt_bboxes[match_idx][:]
            tile_bboxes[:, 0] -= x_1
            tile_bboxes[:, 1] -= y_1
            tile_bboxes[:, 2] -= x_1
            tile_bboxes[:, 3] -= y_1
            tile_bboxes[:, 0] = np.maximum(0, tile_bboxes[:, 0])
            tile_bboxes[:, 1] = np.maximum(0, tile_bboxes[:, 1])
            tile_bboxes[:, 2] = np.minimum(self.tile_size, tile_bboxes[:, 2])
            tile_bboxes[:, 3] = np.minimum(self.tile_size, tile_bboxes[:, 3])
            tile_result['gt_bboxes'] = tile_bboxes
            tile_result['gt_labels'] = tile_lables
            tile_result['gt_masks'] = gt_masks[match_idx].crop(tile_box[0])
        else:
            tile_result.pop('bbox_fields')
            tile_result.pop('mask_fields')
            tile_result.pop('seg_fields')
            tile_result.pop('img_fields')
            tile_result['gt_bboxes'] = []
            tile_result['gt_labels'] = []
            tile_result['gt_masks'] = []

    def __slice_2d(self, length: int) -> List[Tuple[float, float]]:
        """Slices a segment of any length based on the tile size and stride of
        the tiler.

        Args:
            length (int): length of a segment to slice.

        Returns:
            List[Tuple[float, float]]: list of relative start and end points of
                segments resulted from the slicing.
        """
        segments = set()
        for head in range(0, length, self.stride):
            start, end = head, head + self.tile_size
            if end > length:
                start, end = length - self.tile_size, length
            if start < 0:
                start = 0
            segments.add((start, end))
        return list(segments)

    def __tile_boxes_overlap(self, tile_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute overlapping ratio over boxes.

        Args:
            tile_box (np.ndarray): box in shape (1, 4).
            boxes (np.ndarray): boxes in shape (N, 4).

        Returns:
            np.ndarray: overlapping ratio over boxes
        """
        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        width_height = np.minimum(tile_box[:, None, 2:],
                                  boxes[:, 2:]) - np.maximum(
                                      tile_box[:, None, :2], boxes[:, :2])

        width_height = width_height.clip(min=0)  # [N,M,2]
        inter = width_height.prod(2)

        # handle empty boxes
        tile_box_ratio = np.where(inter > 0, inter / box_area,
                                  np.zeros(1, dtype=inter.dtype))
        return tile_box_ratio

    def __multiclass_nms(self, boxes: np.ndarray, scores: np.ndarray,
                         idxs: np.ndarray, iou_threshold: float, max_num: int):
        """NMS for multi-class bboxes.

        Args:
            boxes (np.ndarray):  boxes in shape (N, 4).
            scores (np.ndarray): scores in shape (N, ).
            idxs (np.ndarray):  each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            iou_threshold (float): IoU threshold to be used to suppress boxes
                in tiles' overlap areas.
            max_num (int): if there are more than max_per_img bboxes after
                NMS, only top max_per_img will be kept.

        Returns:
            tuple: tuple: kept dets and indice.
        """
        max_coordinate = boxes.max()
        offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        dets, keep = nms(boxes_for_nms, scores, iou_threshold)
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
        return dets, keep

    @timeit
    def __tile_nms(self, bbox_results: List[List], segm_results,
                   iou_threshold: float, max_per_img: int):
        """NMS after aggregation suppressing duplicate boxes in tile-overlap
        areas.

        Args:
            results (List[List]): image-level prediction
            iou_threshold (float): IoU threshold to be used to suppress boxes
            in tiles' overlap areas.
            max_per_img (int): if there are more than max_per_img bboxes after
                NMS, only top max_per_img will be kept.
        """
        for i, result in enumerate(zip(bbox_results, segm_results)):
            bbox_result, segm_result = result
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0, ), dtype=np.float32)
            labels = np.empty((0, ), dtype=np.float32)
            segms = np.empty((0, ))
            for cls, cls_result in enumerate(zip(bbox_result, segm_result)):
                bbox_cls, segm_cls = cls_result
                bboxes = np.concatenate([bboxes, bbox_cls[:, :4]])
                scores = np.concatenate([scores, bbox_cls[:, 4]])
                labels = np.concatenate([labels, len(bbox_cls) * [cls]])
                segms = np.concatenate([segms, segm_cls])
            _, keep = self.__multiclass_nms(
                bboxes,
                scores,
                labels,
                iou_threshold=iou_threshold,
                max_num=max_per_img)
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            segms = segms[keep]

            bbox_results[i] = bbox2result(np.concatenate(
                [bboxes, scores[:, None]], -1), labels, self.num_classes)

            segm_results[i] = [list(segms[labels == i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.__tiles)

    def __getitem__(self, idx):
        """Get training/test tile.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        result = copy.deepcopy(self.__tiles[idx])
        if osp.isfile(result['tile_path']):
            result['img'] = mmcv.imread(result['tile_path'])
            return result
        dataset_idx = result['dataset_idx']
        x_1, y_1, x_2, y_2 = result['tile_box']
        ori_img = self.__dataset[dataset_idx]['img']
        result['img'] = ori_img[y_1:y_2, x_1:x_2, :]
        return result

    def __process_seg_result(self, seg_result, x1, y1, x2, y2, H, W, img=None):
        if not seg_result:
            return []
        masks = mask_util.decode(seg_result)
        masks = np.pad(masks, ((y1, H - y2), (x1, W - x2), (0, 0)))
        return mask_util.encode(masks)

    @timeit
    def __merge(self, results: List[List]) -> List[List]:
        """Merge/Aggregate tile-level prediction to image-level prediction.

        Args:
            results (list[list | tuple]): Testing tile results of the dataset.

        Returns:
            List[List]: Testing image results of the dataset.
        """
        assert len(results) == len(self.__tiles)

        # BUG: in Segmentation tasks
        if isinstance(results[0], tuple):
            num_classes = len(results[0][0])
            dtype = results[0][0][0].dtype
        elif isinstance(results[0], list):
            num_classes = len(results[0])
            dtype = results[0][0].dtype
        else:
            raise RuntimeError()

        merged_bbox_results = [[
            np.empty((0, 5), dtype=dtype) for _ in range(num_classes)
        ] for _ in range(self.num_images)]

        merged_seg_results = [[[] for _ in range(num_classes)] for _ in range(self.num_images)]

        for n, (result, tile) in enumerate(zip(results, self.__tiles)):
            tile_x1, tile_y1, tile_x2, tile_y2 = tile['tile_box']
            img_idx = tile['dataset_idx']
            img_h, img_w, _ = tile['original_shape_']

            if isinstance(result, tuple):
                bbox_result, mask_result = result
            elif isinstance(result, list):
                bbox_result = result
                mask_result = [[] for _ in range(num_classes)]

            for cls_idx, cls_result in enumerate(
                    zip(bbox_result, mask_result)):
                cls_bbox_result, cls_seg_result = cls_result
                cls_bbox_result[:, 0] = cls_bbox_result[:, 0] + tile_x1
                cls_bbox_result[:, 1] = cls_bbox_result[:, 1] + tile_y1
                cls_bbox_result[:, 2] = cls_bbox_result[:, 2] + tile_x1
                cls_bbox_result[:, 3] = cls_bbox_result[:, 3] + tile_y1

                merged_bbox_results[img_idx][cls_idx] = np.concatenate(
                    (merged_bbox_results[img_idx][cls_idx], cls_bbox_result))
                # FIXME timely
                merged_seg_results[img_idx][
                    cls_idx] += self.__process_seg_result(
                        cls_seg_result, tile_x1, tile_y1, tile_x2, tile_y2,
                        img_h, img_w)

        # run NMS after aggregation suppressing duplicate boxes in
        # overlapping areas
        self.__tile_nms(
            merged_bbox_results,
            merged_seg_results,
            iou_threshold=self.iou_threshold,
            max_per_img=self.max_per_img)

        assert len(merged_bbox_results) == len(merged_seg_results)
        return list(zip(merged_bbox_results, merged_seg_results))

    def evaluate(self, results, **kwargs) -> Dict[str, float]:
        """Evaluation on tiled dataset.

        Evaluate on dataset after merging the tile-level results to image-level
        result.

        Args:
            results (list[list | tuple]): Testing results of the dataset.

        Returns:
            dict[str, float]: evaluation metric.
        """
        merged_results = self.__merge(results)
        return self.__dataset.evaluate(merged_results, **kwargs)
