from typing import Union

import numpy as np
from numpy import ndarray

from .common import Metric, ConfusionMatrix


class MeanIntersectionOverUnionMeter(Metric):
    def __init__(self, num_classes=None, target_classes=None, binary_mod=False, weighted=False,
                 ignore_index=-100, reduce=True, name=None, target_fields=None):
        """Calculates mean intersection over union for a multi-class semantic
        segmentation problem. The meter makes calculations based on confusion matrix

        Keyword arguments:
        :param num_classes: number of classes
        :param target_classes: list of class indexes, which are used for metric calculation.
        Values of a list should be within range [0, num_classes -1]. If set to None or empty list,
        metric is calculated for all classes. If binary_mod is set to True, metric is calculated for both classes.
        :param ignore_index (int, optional): Specifies a target value that is
            ignored and does not contribute to the total iou.
        """
        if binary_mod:
            num_classes = 2
            target_classes = [0, 1]  # in case of binary mode calculate both classes by default
        elif num_classes is None:
            raise ValueError("You must specify number of classes or set `binary_mod` to True")

        if num_classes < 2:
            raise ValueError("Number of classes must be >= 2")
        elif not binary_mod:
            if not target_classes:
                target_classes = list(range(num_classes))

            if max(target_classes) >= num_classes or min(target_classes) < 0:
                bad_vals = [idx for idx in target_classes if idx >= num_classes or idx < 0]
                raise ValueError(f"Values in `target_values` are out of range "
                                 f"[0, {num_classes - 1}], got {bad_vals}")

        if name is None:
            name = f'mIoU_weighted' if weighted else f'mIoU'
        super().__init__(name, target_fields=target_fields)
        self._conf_matrix = ConfusionMatrix(num_classes, False)
        self._num_classes = num_classes
        self._weighted = weighted
        self._ignore_index = ignore_index
        self._reduce = reduce
        self._binary_mod = binary_mod
        self._target_classes = target_classes

    def reset(self):
        super().reset()
        self._conf_matrix.reset()

    def _unify_shapes(self, target, prediction):
        if self._binary_mod:
            if prediction.shape != target.shape:
                raise ValueError('shapes of target and prediction do not match',
                                 target.shape, prediction.shape)
            prediction = prediction > 0
        else:
            # Dimensions check
            if prediction.shape[0] != target.shape[0]:
                raise ValueError('Batch size of target and prediction do not match',
                                 target.shape[0], prediction.shape[0])
            if prediction.ndim == target.ndim + 1:
                prediction = prediction.argmax(1)

            # Dimensions check
            if prediction.shape[1:] != target.shape[1:]:
                raise ValueError('Spatial shapes of target and prediction do not match',
                                 target.shape[1:], prediction.shape[1:])
        return target, prediction

    def _calculate_score(self, conf_matrix):
        tp = np.diagonal(conf_matrix)
        pos_pred = conf_matrix.sum(axis=0)
        pos_gt = conf_matrix.sum(axis=1)

        # Check which classes have elements
        valid_idxs = pos_gt > 0
        ious_valid = valid_idxs & (pos_gt + pos_pred - tp > 0)

        # Calculate intersections over union for each class
        ious = np.zeros((self._num_classes,))
        union = pos_gt[ious_valid] + pos_pred[ious_valid] - tp[ious_valid]
        ious[ious_valid] = tp[ious_valid] / union

        # Calculate mean intersection over union
        iou = self._averaging(ious, ious_valid, pos_gt, conf_matrix)

        return iou

    def _averaging(self, scores, valid_classes, pos_gt, conf_matrix):
        multihot = np.zeros((self._num_classes,), dtype=bool)
        multihot[self._target_classes] = True
        valid_classes = valid_classes & multihot

        if not self._weighted:
            score = np.mean(scores[valid_classes])
        else:
            weights = np.divide(pos_gt, conf_matrix.sum())
            score = scores[valid_classes] @ weights[valid_classes]

        return score

    def _calculate(self, target: ndarray, prediction: ndarray) -> Union[list, ndarray]:
        ious = []
        for true_mask, pred_mask in zip(target, prediction):
            pred_mask = pred_mask.reshape(-1)
            true_mask = true_mask.reshape(-1)

            pred_mask = pred_mask[true_mask != self._ignore_index]
            true_mask = true_mask[true_mask != self._ignore_index]

            conf_matrix = self._conf_matrix.calculate(true_mask, pred_mask)
            ious.append(self._calculate_score(conf_matrix))

        if self._reduce:
            return np.mean(ious)
        else:
            return ious

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        """Calculate IoU metric based on the predicted and target pair.
        Keyword arguments:
        :param prediction: if `binary_mod` is False it can be a (N, *D) tensor of integer values
            between 0 and K-1 or (N, C, *D) tensor of floats values;
            if `binary_mod` is True ir can be a (N, *D) tensor of floats values.
        :param target: Can be a (N, *D) tensor of
            integer values between 0 and K-1.
        """
        target, prediction = self._unify_shapes(target, prediction)
        return self._calculate(target, prediction)

    def update(self, target: ndarray, prediction: ndarray, *args, **kwargs):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        :param prediction: if `binary_mod` is False it can be a (N, *D) tensor of integer values
            between 0 and K-1 or (N, C, *D) tensor of floats values;
            if `binary_mod` is True ir can be a (N, *D) tensor of floats values.
        :param target: Can be a (N, *D) tensor of
            integer values between 0 and K-1.
        """
        target, prediction = self._unify_shapes(target, prediction)

        batch_size = prediction.shape[0]
        value = self._calculate(target, prediction) * batch_size
        self.mean = (self.n * self.mean + value) / (self.n + batch_size)
        self.n += batch_size


class IntersectionOverUnionMeter(MeanIntersectionOverUnionMeter):
    def __init__(self, target_class, binary_mod=False, ignore_index=-100,
                 reduce=True, name=None, target_fields=None):
        """Calculates intersection over union for a certain class for a semantic segmentation problem.
        The meter makes calculations based on confusion matrix
        Keyword arguments:
        :param target_class: index of class for which the IoU will be calculated.
        :param ignore_index (int, optional): Specifies a target value that is
            ignored and does not contribute to the total iou.
        """
        if name is None:
            name = f'IoU_class={target_class}'

        super().__init__(2, ignore_index=ignore_index, binary_mod=binary_mod,
                         reduce=reduce, name=name, target_fields=target_fields)
        self.target_class = target_class

    def _calculate_score(self, conf_matrix):
        tp = conf_matrix[1, 1]
        fs = conf_matrix[1, 0] + conf_matrix[0, 1]
        if tp + fs == 0:
            return 1
        else:
            return tp / (tp + fs)

    def _unify_shapes(self, target, prediction):
        target, prediction = super()._unify_shapes(target, prediction)
        target = (target == self.target_class)
        prediction = (prediction == self.target_class)
        return target, prediction


class MeanDiceMeter(MeanIntersectionOverUnionMeter):
    def __init__(self, num_classes=None, target_classes=None, binary_mod=False, weighted=False,
                 ignore_index=-100, reduce=True, name=None, target_fields=None):
        """Calculates mean Dice similarity coefficient for a multi-class semantic
        segmentation problem. The meter makes calculations based on confusion matrix

        Keyword arguments:
        :param num_classes: number of classes
        :param target_classes: list of class indexes, which are used for metric calculation.
        Values of a list should be within range [0, num_classes -1]. If set to None or empty list,
        metric is calculated for all classes. If binary_mod is set to True, metric is calculated for both classes.
        :param ignore_index (int, optional): Specifies a target value that is
            ignored and does not contribute to the total iou.
        """
        if name is None:
            name = f'mDice'

        super().__init__(num_classes=num_classes, target_classes=target_classes,
                         ignore_index=ignore_index, binary_mod=binary_mod, weighted=weighted,
                         reduce=reduce, name=name, target_fields=target_fields)

    def _calculate_score(self, conf_matrix):
        tp = np.diagonal(conf_matrix)
        pos_pred = conf_matrix.sum(axis=0)
        pos_gt = conf_matrix.sum(axis=1)

        # Check which classes have elements
        valid_idxs = pos_gt > 0
        if 0 <= self._ignore_index < self._num_classes:
            valid_idxs[self._ignore_index] = False
        valid_classes = valid_idxs & (pos_gt + pos_pred - tp > 0)

        # Calculate intersections over union for each class
        dice_scores = np.zeros((self._num_classes,))
        dice_scores[valid_classes] = 2 * tp[valid_classes] / (pos_gt[valid_classes] + pos_pred[valid_classes])

        # Calculate mean intersection over union
        mean_dice = self._averaging(dice_scores, valid_classes, pos_gt, conf_matrix)

        return mean_dice


class DiceMeter(IntersectionOverUnionMeter):
    def __init__(self, target_class, binary_mod=False, ignore_index=-100,
                 reduce=True, name=None, target_fields=None):
        """Calculates Dice similarity coefficient for a certain class for a semantic segmentation problem.
        The meter makes calculations based on confusion matrix
        Keyword arguments:
        :param target_class: index of class for which the IoU will be calculated.
        :param ignore_index (int, optional): Specifies a target value that is
            ignored and does not contribute to the total iou.
        """
        if name is None:
            name = f'Dice_class={target_class}'

        super().__init__(2, ignore_index=ignore_index, binary_mod=binary_mod,
                         reduce=reduce, name=name, target_fields=target_fields)
        self.target_class = target_class

    def _calculate_score(self, conf_matrix):
        tp = conf_matrix[1, 1]
        fs = conf_matrix[1, 0] + conf_matrix[0, 1]
        if tp + fs == 0:
            return 1
        else:
            return 2 * tp / (fs + 2 * tp)
