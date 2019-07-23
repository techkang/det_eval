import os
import tempfile

import cv2
import numpy as np
from shapely.geometry import Polygon


class DetEval(object):
    r""" Det Eval Scripts

    """

    def __init__(self, tr=0.7, tp=0.6, min_polygon_threshold=30, detection_filter=0.5):
        self.FSC_K = 0.8
        self.TR = tr
        self.TP = tp
        self.K = 2
        self.MIN_POLYGON_THRESHOLD = min_polygon_threshold
        self.detection_filter = detection_filter

    def get_polygon(self, heatmap):
        _, polygons, _ = cv2.findContours(heatmap.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = np.array([i[:, 0] for i in polygons if cv2.contourArea(i) > self.MIN_POLYGON_THRESHOLD])
        return polygons

    def prepare_data(self, file):
        """ read data from file
        :param file: image file
        :return: pred(detection), gt, not_cares, all are polygons.
        """
        maps = np.load(file)
        pred, gt, not_care = maps[-3:]
        return self.get_polygon(pred), self.get_polygon(gt), self.get_polygon(not_care)

    @staticmethod
    def area(poly):
        polygon = Polygon(poly)
        return float(polygon.area)

    @staticmethod
    def area_of_intersection(p1, p2):
        p1 = Polygon(p1).buffer(0)
        p2 = Polygon(p2).buffer(0)
        return float(p1.intersection(p2).area)

    @staticmethod
    def area_of_union(p1, p2):
        p1 = Polygon(p1).buffer(0)
        p2 = Polygon(p2).buffer(0)
        return float(p1.union(p2).area)

    def iou(self, p1, p2):
        return self.area_of_intersection(p1, p2) / max(1., (self.area_of_union(p1, p2)))

    def intersect_over_p1(self, p1, p2):
        return self.area_of_intersection(p1, p2) / max(1., self.area(p1))

    def detection_filtering(self, detections, not_cares):
        result_detection = []

        for det_id, det in enumerate(detections):
            flag = True
            for gt_id, gt in enumerate(not_cares):
                det_gt_iou = self.intersect_over_p1(det, gt)
                if det_gt_iou > self.detection_filter:
                    flag = False
                    break
            if flag:
                result_detection.append(det)

        return result_detection

    def sigma_calculation(self, det, gt):
        """
        sigma = intersection_area / gt_area
        """
        return np.round((self.area_of_intersection(det, gt) / self.area(gt)), 2)

    def tau_calculation(self, det, gt):
        """
        tau = intersection_area / det_area
        """
        return np.round((self.area_of_intersection(det, gt) / self.area(det)), 2)

    def one_to_one(self, local_sigma_table, local_tau_table, gt_flag, det_flag, num_gt):
        local_accumulative_precision = 0
        local_accumulative_recall = 0
        for gt_id in range(num_gt):
            if np.sum(local_sigma_table[gt_id, :] > self.TR) != 1:
                continue
            detection_index = np.where(local_sigma_table[gt_id] > self.TR)[0]
            if np.sum(local_sigma_table[:, detection_index] > self.TR) != 1:
                continue
            if np.sum(local_tau_table[gt_id, :] > self.TP) != 1:
                continue
            if np.sum(local_tau_table[:, detection_index] > self.TP) != 1:
                continue

            local_accumulative_recall += 1.0
            local_accumulative_precision += 1.0

            gt_flag[gt_id] = 1
            det_flag[detection_index] = 1
        return local_accumulative_precision, local_accumulative_recall

    # one gt covers many detections
    def one_to_many(self, local_sigma_table, local_tau_table, gt_flag, det_flag, num_gt):
        local_accumulative_precision = 0
        local_accumulative_recall = 0
        for gt_id in range(num_gt):
            # skip the following if the groundtruth was matched
            if gt_flag[gt_id] > 0:
                continue

            detection_indices = np.where(local_tau_table[gt_id] > self.TP)[0]
            if len(detection_indices) < self.K:
                continue
            if np.sum(local_sigma_table[gt_id, detection_indices]) < self.TR:
                continue
            gt_flag[gt_id] = 1
            det_flag[detection_indices] = 1

            local_accumulative_recall += self.FSC_K
            local_accumulative_precision += len(detection_indices) * self.FSC_K

        return local_accumulative_precision, local_accumulative_recall

    # one detection covers many groundtruths
    def many_to_one(self, local_sigma_table, local_tau_table, gt_flag, det_flag, num_det):
        local_accumulative_precision = 0
        local_accumulative_recall = 0
        for det_id in range(num_det):
            # skip the following if the detection was matched
            if det_flag[det_id] > 0:
                continue

            gt_indices = np.where(local_sigma_table[:, det_id] > self.TR)[0]
            if len(gt_indices) < self.K:
                continue
            if np.sum(local_tau_table[gt_indices, det_id]) < self.TP:
                continue
            gt_flag[gt_indices] = 1
            det_flag[det_id] = 1

            local_accumulative_recall += len(gt_indices) * self.FSC_K
            local_accumulative_precision += self.FSC_K

        return local_accumulative_precision, local_accumulative_recall

    def eval(self, path):
        files = os.listdir(path)
        global_precision_recall = np.array([0., 0.])
        total_gt_num = 0
        total_det_num = 0
        for file in files:
            detections, groundtruths, not_cares = self.prepare_data(os.path.join(path, file))
            # filters detections overlapping with not_care area
            detections = self.detection_filtering(detections, not_cares)

            sigma_table = np.zeros((len(groundtruths), len(detections)))
            tau_table = np.zeros((len(groundtruths), len(detections)))

            for gt_id, gt in enumerate(groundtruths):
                if len(detections) > 0:
                    for det_id, det in enumerate(detections):
                        sigma_table[gt_id, det_id] = self.sigma_calculation(det, gt)
                        tau_table[gt_id, det_id] = self.tau_calculation(det, gt)

            num_gt = sigma_table.shape[0]
            num_det = sigma_table.shape[1]

            total_gt_num += num_gt
            total_det_num += num_det

            gt_flag = np.zeros(num_gt)
            det_flag = np.zeros(num_det)
            local_precision_recall = np.array([0., 0.])

            local_precision_recall += self.one_to_one(sigma_table, tau_table, gt_flag, det_flag, num_gt)
            local_precision_recall += self.one_to_many(sigma_table, tau_table, gt_flag, det_flag, num_gt)
            local_precision_recall += self.many_to_one(sigma_table, tau_table, gt_flag, det_flag, num_det)

            global_precision_recall += local_precision_recall

        # if total_gt_num is zero, global_precision_recall[1] must be 0, max function is used to avoid ZeroDivisionError
        recall = global_precision_recall[1] / max(1, total_gt_num)
        precision = global_precision_recall[0] / max(1, total_det_num)
        f_score = 2 * precision * recall / max(1, (precision + recall))

        return precision, recall, f_score


if __name__ == '__main__':
    vertical_step = np.array([20, 0])
    poly1 = np.array([[10, 10], [10, 80], [20, 80], [20, 10]])
    poly2 = poly1 + vertical_step
    poly2_1 = np.array([[30, 10], [30, 40], [40, 40], [40, 10]])
    poly2_2 = poly2_1 + np.array([0, 40])
    poly3 = poly2 + vertical_step
    poly3_1 = poly2_1 + vertical_step
    poly3_2 = poly2_2 + vertical_step
    poly4 = poly3 + vertical_step
    det = [poly1, poly2, poly3_1, poly3_2, poly4]
    gt = [poly1, poly2_1, poly2_2, poly3]
    not_care = [poly4]
    det = cv2.fillPoly(np.zeros((100, 100)), det, 1)
    gt = cv2.fillPoly(np.zeros((100, 100)), gt, 1)
    not_care = cv2.fillPoly(np.zeros((100, 100)), not_care, 1)
    det_eval = DetEval()
    with tempfile.TemporaryDirectory() as temp_path:
        np.save(os.path.join(temp_path, 'image'), np.array([det, gt, not_care]))
        print('precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}'.format(*det_eval.eval(path=temp_path)))
