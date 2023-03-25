# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.evaluation import bbox_overlaps 
from mmengine.logging import print_log
from multiprocessing import get_context
from terminaltables import AsciiTable


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5, 
                 area_ranges=None,
                 use_legacy_coordinate=False):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Defaults to None
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        area_ranges (list[tuple], optional): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Defaults to None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp
    
    ious = bbox_overlaps(
        det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (
                        bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['bboxes']) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts.append(torch.zeros((0, 4), dtype=torch.float64))
            cls_gts_ignore.append(torch.zeros((0, 4), dtype=torch.float64))
    return cls_dets, cls_gts, cls_gts_ignore


def eval_f1score(det_results,
                       annotations,
                       scale_ranges=None,
                       iou_thr=0.5,
                       dataset=None,
                       logger=None,
                       nproc=4,
                       use_legacy_coordinate=False):
    assert len(det_results) == len(annotations)

    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)],
                [use_legacy_coordinate for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
                    bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        # tp = np.hstack(tp)[:, sort_inds]
        # fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        # tp = np.cumsum(tp, axis=1)
        # fp = np.cumsum(fp, axis=1)
        tp = np.hstack(tp).sum(axis=1)
        fp = np.hstack(fp).sum(axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0].item()
            precisions = precisions[0].item()
            num_gts = num_gts.item()
        if num_gts > 0:
            f1_score = (2 * precisions * recalls + eps) / (precisions + recalls + eps)
        else:
            f1_score = 0

        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'f1_score': f1_score,
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_f1_score = np.vstack(
            [cls_result['f1_score'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_f1_score = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_f1_score.append(all_f1_score[all_num_gts[:, i] > 0,
                                                  i].mean())
            else:
                mean_f1_score.append(0.0)
    else:
        f1_scores = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                f1_scores.append(cls_result['f1_score'])

        mean_f1_score = np.array(f1_scores).mean().item() if f1_scores else 0.0

    print_f1_score_summary(
        mean_f1_score, eval_results, dataset, area_ranges, logger=logger)

    return mean_f1_score, eval_results

def print_f1_score_summary(mean_f1_score,
                           results,
                           dataset=None,
                           scale_ranges=None,
                           logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str, optional): Dataset name or dataset classes.
        scale_ranges (list[tuple], optional): Range of scales to be evaluated.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['f1_score'], np.ndarray):
        num_scales = len(results[0]['f1_score'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    f1_scores = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        recalls[:, i] = np.array(cls_result['recall'], ndmin=2)
        precisions[:, i] = np.array(cls_result['precision'], ndmin=2)
        f1_scores[:, i] = cls_result['f1_score']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_f1_score, list):
        mean_f1_score = [mean_f1_score]

    header = ['class', 'gts', 'dets', 'recall', 'precision', 'f1_score']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}', f'{f1_scores[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(
            ['mean_f1_score', '', '', '', '', f'{mean_f1_score[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
