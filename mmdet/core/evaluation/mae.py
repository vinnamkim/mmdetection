import math
from mmdet.core import bbox
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error


class ScoreMetric:
    """ Score Metric """

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

        if math.isnan(value):
            raise ValueError(
                "The value of a ScoreMetric is not allowed to be NaN.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreMetric):
            return False
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        return f"ScoreMetric(name=`{self.name}`, score=`{self.value}`)"

    @staticmethod
    def type():
        return "score"


class CurveMetric:
    """ Curve Metric """

    def __init__(self, name: str, ys: List[float], xs: Optional[List[float]] = None):
        self.name = name
        self.__ys = ys
        if xs is not None:
            if len(xs) != len(self.__ys):
                raise ValueError(
                    f"Curve error must contain the same length for x and y: ({len(xs)} vs {len(self.ys)})"
                )
            self.__xs = xs
        else:
            # if x values are not provided, set them to the 1-index of the y values
            self.__xs = list(range(1, len(self.__ys) + 1))

    @property
    def ys(self) -> List[float]:
        """
        Returns the list of floats on y-axis.
        """
        return self.__ys

    @property
    def xs(self) -> List[float]:
        """
        Returns the list of floats on x-axis.
        """
        return self.__xs

    def __repr__(self):
        return (
            f"CurveMetric(name=`{self.name}`, ys=({len(self.ys)} values), "
            f"xs=({len(self.xs) if self.xs is not None else 'None'} values))"
        )

    @staticmethod
    def type():
        return "curve"


class _AggregatedResults:
    def __init__(
        self,
        mae_curve: Dict[str, List[float]],
        relative_mae_curve: Dict[str, List[float]],
        all_classes_mae_curve: List[float],
        all_classes_relative_mae_curve: List[float],
        best_y_pred,
        best_y_true,
        best_mae: float,
        best_relative_mae: float,
        best_threshold: float,
    ):
        self.relative_mae_curve = relative_mae_curve
        self.mae_curve = mae_curve
        self.all_classes_mae_curve = all_classes_mae_curve
        self.all_classes_relative_mae_curve = all_classes_relative_mae_curve
        self.best_mae = best_mae
        self.best_relative_mae = best_relative_mae
        self.best_threshold = best_threshold
        self.best_y_pred = best_y_pred
        self.best_y_true = best_y_true


class _ResultCounters:
    def __init__(self, difference):
        self.difference = difference


class _Metrics:
    def __init__(
        self, mae: float, relative_mae: float, y_pred: np.ndarray, y_true: np.ndarray
    ):
        self.mae = mae
        self.relative_mae = relative_mae
        self.y_pred = y_pred
        self.y_true = y_true


class _OverallResults:
    def __init__(
        self,
        per_confidence: _AggregatedResults,
        best_mae_per_class: Dict[str, float],
        best_relative_mae_per_class: Dict[str, float],
        best_mae: float,
        best_relative_mae: float,
    ):
        self.per_confidence = per_confidence
        self.best_mae_per_class = best_mae_per_class
        self.best_relative_mae_per_class = best_relative_mae_per_class
        self.best_relative_mae = best_relative_mae
        self.best_mae = best_mae


class MAE:
    """ Mean Absolute Error Metric
    Returns:
        _type_: _description_
    """
    box_score_index = 0
    box_class_index = 1

    def __init__(self, cocoDt, cocoGt, vary_confidence_threshold: bool = False):
        confidence_range = [0.025, 1.0, 0.025]
        confidence_values = list(np.arange(*confidence_range))
        prediction_boxes_per_image = self.prepare(cocoDt)
        ground_truth_boxes_per_image = self.prepare(cocoGt)
        assert len(prediction_boxes_per_image) == len(
            ground_truth_boxes_per_image)
        classes = {v["id"]: v["name"] for k, v in cocoGt.cats.items()}

        result = self.evaluate_detections(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            predicted_boxes_per_image=prediction_boxes_per_image,
            confidence_range=confidence_range,
            classes=classes,
            img_ids=cocoGt.getImgIds(),
        )

        self._mae = ScoreMetric(
            name="mean-absolute-error", value=result.best_mae)

        self._relative_mae = ScoreMetric(
            name="relative-mean-absolute-error", value=result.best_relative_mae
        )

        mae_per_label: Dict[str, ScoreMetric] = {}
        relative_mae_per_label: Dict[str, ScoreMetric] = {}
        for class_idx, class_name in classes.items():
            mae_per_label[class_name] = ScoreMetric(
                name=class_name, value=result.best_mae_per_class[class_name]
            )
            relative_mae_per_label[class_name] = ScoreMetric(
                name=class_name, value=result.best_relative_mae_per_class[class_name]
            )
        self._mae_per_label = mae_per_label
        self._relative_mae_per_label = relative_mae_per_label

        self._mae_per_confidence: Optional[CurveMetric] = None
        self._best_confidence_threshold: Optional[ScoreMetric] = None

        if vary_confidence_threshold:
            mae_per_confidence = CurveMetric(
                name="MAE per confidence",
                xs=confidence_values,
                ys=result.per_confidence.all_classes_mae_curve,
            )
            best_confidence_threshold = ScoreMetric(
                name="Optimal confidence threshold",
                value=result.per_confidence.best_threshold,
            )
            self._mae_per_confidence = mae_per_confidence
            self._best_confidence_threshold = best_confidence_threshold

    def prepare(self, cocoAPI) -> OrderedDict:
        new_annotations = OrderedDict()
        for image_id, bboxes in cocoAPI.imgToAnns.items():
            new_annotations[image_id] = []
            for b in bboxes:
                x1, y1, w, h = b["bbox"]
                score = b["score"] if "score" in b else 1.0
                new_annotations[image_id].append(
                    [x1, y1, x1 + w, y1 + h, score, b["category_id"]]
                )
        for image_id in cocoAPI.getImgIds():
            if image_id not in new_annotations:
                new_annotations[image_id] = []
        return new_annotations

    def evaluate_detections(
        self,
        ground_truth_boxes_per_image: Dict,
        predicted_boxes_per_image: Dict,
        classes: Dict[int, str],
        img_ids,
        confidence_range: List[float] = None,
    ):

        best_mae_per_class = {}
        best_relative_mae_per_class = {}

        if confidence_range is None:
            confidence_range = [0.025, 1.0, 0.025]

        results_per_confidence = self.get_results_per_confidence(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            predicted_boxes_per_image=predicted_boxes_per_image,
            classes=classes,
            confidence_range=confidence_range,
            img_ids=img_ids,
        )

        best_mae = results_per_confidence.best_mae
        best_relative_mae = results_per_confidence.best_relative_mae

        for _, class_name in classes.items():
            idx = np.argmin(results_per_confidence.mae_curve[class_name])
            best_mae_per_class[class_name] = results_per_confidence.mae_curve[
                class_name
            ][idx]
            best_relative_mae_per_class[
                class_name
            ] = results_per_confidence.relative_mae_curve[class_name][idx]

        result = _OverallResults(
            results_per_confidence,
            best_mae_per_class=best_mae_per_class,
            best_mae=best_mae,
            best_relative_mae_per_class=best_relative_mae_per_class,
            best_relative_mae=best_relative_mae,
        )
        return result

    def get_results_per_confidence(
        self,
        ground_truth_boxes_per_image: Dict,
        predicted_boxes_per_image: Dict,
        classes: Dict[int, str],
        confidence_range: List[float],
        img_ids,
        all_classes_name: str = "All Classes",
    ) -> _AggregatedResults:

        result = _AggregatedResults(
            mae_curve={class_name: [] for _, class_name in classes.items()},
            relative_mae_curve={class_name: []
                                for _, class_name in classes.items()},
            all_classes_mae_curve=[],
            all_classes_relative_mae_curve=[],
            best_y_pred=[],
            best_y_true=[],
            best_mae=9999.0,
            best_relative_mae=9999.0,
            best_threshold=0.1,
        )

        for confidence_threshold in np.arange(*confidence_range):
            result_point = self.evaluate_classes(
                ground_truth_boxes_per_image=ground_truth_boxes_per_image,
                predicted_boxes_per_image=MAE.filter_confidence(
                    predicted_boxes_per_image, confidence_threshold
                ),
                classes=classes,
                img_ids=img_ids,
            )
            all_classes_mae = result_point[all_classes_name].mae
            all_classes_relative_mae = result_point[all_classes_name].relative_mae
            y_true = result_point[all_classes_name].y_true
            y_pred = result_point[all_classes_name].y_pred
            result.all_classes_mae_curve.append(all_classes_mae)
            result.all_classes_relative_mae_curve.append(
                all_classes_relative_mae)
            for _, class_name in classes.items():
                result.mae_curve[class_name].append(
                    result_point[class_name].mae)
                result.relative_mae_curve[class_name].append(
                    result_point[class_name].relative_mae
                )

            if all_classes_mae < result.best_mae:
                result.best_mae = all_classes_mae
                result.best_relative_mae = all_classes_relative_mae
                result.best_threshold = confidence_threshold
                result.best_y_pred = y_pred
                result.best_y_true = y_true
        return result

    @staticmethod
    def evaluate_classes(
        ground_truth_boxes_per_image: Dict,
        predicted_boxes_per_image: Dict,
        classes: Dict[int, str],
        img_ids,
    ) -> Dict[str, _Metrics]:

        all_classes_name = "All Classes"
        result: Dict[str, _Metrics] = {}

        diffs = []
        relative_ae_diffs = []
        y_preds = []
        y_trues = []
        for class_idx, class_name in classes.items():
            class_ground_truth_boxes_per_image = MAE.filter_class(
                ground_truth_boxes_per_image, class_idx
            )

            class_predicted_boxes_per_image = MAE.filter_class(
                predicted_boxes_per_image,
                class_idx,
            )

            metrics = MAE.get_mae(
                class_ground_truth_boxes_per_image,
                class_predicted_boxes_per_image,
                img_ids=img_ids,
            )

            y_preds.extend(list(metrics.y_pred))
            y_trues.extend(list(metrics.y_true))
            diff = list(np.abs(metrics.y_pred - metrics.y_true))
            relative_ae_diff = list(diff / metrics.y_true)
            result[class_name] = metrics
            diffs.extend(diff)
            relative_ae_diffs.extend(relative_ae_diff)

        # for all classes
        result[all_classes_name] = _Metrics(
            np.average(diffs),
            np.average(relative_ae_diffs),
            y_pred=y_preds,
            y_true=y_trues,
        )
        return result

    @staticmethod
    def get_mae(
        class_ground_truth_boxes_per_image: Dict,
        class_predicted_boxes_per_image: Dict,
        img_ids: List[int],
    ) -> Tuple[_Metrics, _ResultCounters]:

        y_pred = np.array(
            [len(class_predicted_boxes_per_image[idx]) for idx in img_ids]
        )
        y_true = np.array(
            [len(class_ground_truth_boxes_per_image[idx]) for idx in img_ids]
        )

        keep_indices = y_true != 0
        y_pred = y_pred[keep_indices]
        y_true = y_true[keep_indices]

        diff = np.abs(y_pred - y_true)
        relative_ae = list(diff / y_true)

        results = _Metrics(
            mae=np.average(diff),
            relative_mae=np.average(relative_ae),
            y_pred=y_pred,
            y_true=y_true,
        )
        return results

    @staticmethod
    def filter_class(boxes_per_image: Dict, class_idx: int) -> OrderedDict:
        """
        Filters boxes to only keep members of one class
        :param boxes_per_image:
        :param class_name:
        :return:
        """
        filtered_boxes_per_image = OrderedDict()
        for image_id, boxes in boxes_per_image.items():
            filtered_boxes = []
            for box in boxes:
                if box[MAE.box_class_index] == class_idx:
                    filtered_boxes.append(box)
            filtered_boxes_per_image[image_id] = filtered_boxes
        return filtered_boxes_per_image

    @staticmethod
    def filter_confidence(
        boxes_per_image: Dict, confidence_threshold: float
    ) -> OrderedDict:
        """
        Filters boxes to only keep ones with higher confidence than a given confidence threshold
        :param boxes_per_image: shape List[List[[Tuple[float, str]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
        :param confidence_threshold:
        :return:
        """
        filtered_boxes_per_image = OrderedDict()
        for image_id, boxes in boxes_per_image.items():
            filtered_boxes = []
            for box in boxes:
                if float(box[MAE.box_score_index]) > confidence_threshold:
                    filtered_boxes.append(box)
            filtered_boxes_per_image[image_id] = filtered_boxes
        return filtered_boxes_per_image

    @property
    def mae(self) -> ScoreMetric:
        return self._mae

    @property
    def mae_per_label(self) -> Dict[str, ScoreMetric]:
        return self._mae_per_label

    @property
    def relative_mae(self) -> ScoreMetric:
        return self._relative_mae

    @property
    def relative_mae_per_label(self) -> Dict[str, ScoreMetric]:
        return self._relative_mae_per_label

    @property
    def best_confidence_threshold(self) -> Optional[ScoreMetric]:
        return self._best_confidence_threshold


class CustomMAE(MAE):
    def __init__(self, pred_bboxes, gt_bboxes, vary_confidence_threshold: bool = False, labels: list = [], 
                 img_ids: list = []):
        confidence_range = [0.025, 1.0, 0.025]
        confidence_values = list(np.arange(*confidence_range))
        prediction_boxes_per_image = self.prepare(pred_bboxes, img_ids)
        ground_truth_boxes_per_image = self.prepare(gt_bboxes, img_ids)
        assert len(prediction_boxes_per_image) == len(
            ground_truth_boxes_per_image)
        classes = {i: v for i, v in enumerate(labels)}
        result = self.evaluate_detections(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            predicted_boxes_per_image=prediction_boxes_per_image,
            confidence_range=confidence_range,
            classes=classes,
            img_ids=img_ids,
        )

        self._mae = ScoreMetric(
            name="mean-absolute-error", value=result.best_mae)

        self._relative_mae = ScoreMetric(
            name="relative-mean-absolute-error", value=result.best_relative_mae
        )

        mae_per_label: Dict[str, ScoreMetric] = {}
        relative_mae_per_label: Dict[str, ScoreMetric] = {}
        for class_idx, class_name in classes.items():
            mae_per_label[class_name] = ScoreMetric(
                name=class_name, value=result.best_mae_per_class[class_name]
            )
            relative_mae_per_label[class_name] = ScoreMetric(
                name=class_name, value=result.best_relative_mae_per_class[class_name]
            )
        self._mae_per_label = mae_per_label
        self._relative_mae_per_label = relative_mae_per_label

        self._mae_per_confidence: Optional[CurveMetric] = None
        self._best_confidence_threshold: Optional[ScoreMetric] = None

        if vary_confidence_threshold:
            mae_per_confidence = CurveMetric(
                name="MAE per confidence",
                xs=confidence_values,
                ys=result.per_confidence.all_classes_mae_curve,
            )
            best_confidence_threshold = ScoreMetric(
                name="Optimal confidence threshold",
                value=result.per_confidence.best_threshold,
            )
            self._mae_per_confidence = mae_per_confidence
            self._best_confidence_threshold = best_confidence_threshold

    def prepare(self, results, img_ids) -> OrderedDict:
        new_annotations = OrderedDict()
        for image_id in img_ids:
            if image_id not in new_annotations:
                new_annotations[image_id] = []
        for result in results:
            image_id = result['image_id']
            score = result["score"] if "score" in result else 1.0
            new_annotations[image_id].append([score, result["category_id"]])
        return new_annotations
