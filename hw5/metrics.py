#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import metrics
import os
import cv2


def accuracy(y_true, y_pred):
    '''
    Accuracy metric.
    Input:
        y_true - list, ground truth classes
        y_pred - list, predicted classes
    Output:
        acc - float, accuracy
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ones = np.ones_like(y_true)
    zeros = np.zeros_like(y_true)
    return np.sum(np.where(y_pred == y_true, ones, zeros)) / len(y_true)


def precision(y_true, y_pred):
    '''
    Precision metric.
    Input:
        y_true - list, ground truth classes
        y_pred - list, predicted classes
    Output:
        precisions - array-like, precisions for each class
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ones = np.ones_like(y_true)
    zeros = np.zeros_like(y_true)
    classes = np.unique(y_true)
    precisions = []
    buf_pred = np.where(y_pred == y_true, y_pred, np.ones_like(y_true) * 0.5)
    buf_unpred = np.where(y_pred != y_true, y_pred, np.ones_like(y_true) * 0.5)
    for cl in classes:
        tp = np.sum(np.where(buf_pred == cl, ones, zeros))
        fp = np.sum(np.where(buf_unpred == cl, ones, zeros))
        precisions.append(tp / (tp + fp))
    return precisions


def recall(y_true, y_pred):
    '''
    Recall metric.
    Input:
        y_true - list, ground truth classes
        y_pred - list, predicted classes
    Output:
        recalls - array-like, recalls for each class
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ones = np.ones_like(y_true)
    zeros = np.zeros_like(y_true)
    classes = np.unique(y_true)
    recalls = []
    buf_pred = np.where(y_pred == y_true, y_pred, np.ones_like(y_true) * 0.5)
    buf_unpred = np.where(y_pred != y_true, y_true, np.ones_like(y_true) * 0.5)
    for cl in classes:
        tp = np.sum(np.where(buf_pred == cl, ones, zeros))
        fn = np.sum(np.where(buf_unpred == cl, ones, zeros))
        recalls.append(tp / (tp + fn))
    return recalls


def f1(y_true, y_pred):
    '''
    F1 metric.
    Input:
        y_true - list, ground truth classes
        y_pred - list, predicted classes
    Output:
        f1_measure - array-like, f1-measures for each class
    '''
    prec = np.array(precision(y_true, y_pred))
    rec = np.array(recall(y_true, y_pred))
    f1_measure = 2 * (prec * rec / (prec + rec))
    return f1_measure


def rmse(y_true, y_pred):
    '''
    RMSE metric.
    Input:
        y_true - list, ground truth values
        y_pred - list, predicted values
    Output:
        metric - float, RMSE
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    metric = np.sqrt(np.sum((y_true - y_pred) ** 2) / y_true.shape[0])
    return metric


def mae(y_true, y_pred):
    '''
    MAE metric.
    Input:
        y_true - list, ground truth values
        y_pred - list, predicted values
    Output:
        metric - float, MAE
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    metric = np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]
    return metric


def iou(bb_true, bb_pred):
    '''
    Intersection over union metric.
    Input:
        bb_true - array-like, ground truth bounding box, contain 2 tuples
        [(x1, y1), (x2, y2)]
        bb_pred - array-like, predicted bounding box, similar to bb_true
    Output:
        iou_metr - float, IOU metric
    '''
    # find intersection coords
    # coords placed as it used in opencv
    inter = [(max(bb_true[0][0], bb_pred[0][0]),
              min(bb_true[0][1], bb_pred[0][1])),
             (min(bb_true[1][0], bb_pred[1][0]),
              max(bb_true[1][1], bb_pred[1][1]))]
    s_true = (bb_true[1][0] - bb_true[0][0] + 1) * (bb_true[0][1] - bb_true[1][1] + 1)
    s_pred = (bb_pred[1][0] - bb_pred[0][0] + 1) * (bb_pred[0][1] - bb_pred[1][1] + 1)
    s_inter = max(0, inter[1][0] - inter[0][0] + 1) * max(0, inter[0][1] - inter[1][1] + 1)
    iou_metr = s_inter / (s_true + s_pred - s_inter)
    return iou_metr


def AP(gts, prs, threshold=0.3, mode='all'):
    '''
    Average precision metric. Example of gts and prs arrays in function
    readfiles_for_AP. All objects must be from 1 class.
    Input:
        gts - array-like, ground truth array of dicts with keys: 'class', 'bb'
        prs - array-like, predicted array of dicts with keys: 'class', 'bb',
        'conf', 'iou'
        threshold - float, threshold for IOU - if IOU of predicted bounding box
        greater than threshold, it's considered as TP, else as FP
        mode - str, '11' or 'all'. '11' for 11-point interpolation, 'all' for
        calculation AP with all points.
    Output:
        ap - float, average precision
    '''
    # all examples from 1 class
    for i, gt in enumerate(gts):
        pr = prs[i]
        for gtr in gt:
            max_iou = 0
            pred_ind = None
            for pred in pr:
                buf_iou = iou(gtr['bb'], pred['bb'])
                if buf_iou > pred['iou'] and buf_iou > max_iou:
                    pred_ind = pr.index(pred)
                    max_iou = buf_iou
            if pred_ind is not None:
                pr[pred_ind]['iou'] = max_iou
                pred_ind = None
    # calculating TP and FP, and obtain points of precision-recall curve
    tps = []
    precisions = []
    recalls = []
    len_gts = np.sum([len(x) for x in gts])
    predicted = []
    for pr in prs:
        predicted += pr
    predicted = sorted(predicted, key=lambda k: k['conf'], reverse=True)
    for pred in predicted:
        if pred['iou'] > threshold:
            tps.append(1)
        else:
            tps.append(0)
        precisions.append(np.sum(tps) / len(tps))
        recalls.append(np.sum(tps) / len_gts)
    # calculating AP
    if mode == 'all':
        ap = 0
        prev_rec = 0
        prev_prec_ind = 0
        while(1):
            max_prec = np.max(precisions[prev_prec_ind:])
            ind = precisions[prev_prec_ind:].index(max_prec) + prev_prec_ind
            rec = recalls[ind]
            ap += max_prec * (rec - prev_rec)
            prev_rec = rec
            prev_prec_ind = ind + 1
            if prev_prec_ind >= len(precisions) - 1:
                break
    elif mode == '11':
        ap = 0
        points = np.linspace(0, 1, 11)
        for p in points:
            rec_buf = np.where(np.array(recalls) >= p, np.ones_like(recalls),
                               np.zeros_like(recalls))
            if 1 in rec_buf:
                rec_ind = list(rec_buf).index(1)
                cur_prec = np.max(precisions[rec_ind:])
            else:
                cur_prec = 0
            ap += cur_prec
        ap /= 11
    return ap


def mAP(gts, prs, threshold=0.3, mode='all'):
    '''
    Mean average precision metric. Example of gts and prs arrays in function
    readfiles_for_AP.
    Input:
        gts - array-like, ground truth array of dicts with keys: 'class', 'bb'
        prs - array-like, predicted array of dicts with keys: 'class', 'bb',
        'conf', 'iou'
        threshold - float, threshold for IOU - if IOU of predicted bounding box
        greater than threshold, it's considered as TP, else as FP
        mode - str, '11' or 'all'. '11' for 11-point interpolation, 'all' for
        calculation AP with all points.
    Output:
        map - float, mean average precision
    '''
    classes = []
    aps = []
    for gt in gts:
        for gtr in gt:
            classes.append(gtr['class'])
    classes = set(classes)
    for cl in classes:
        cur_gts = []
        cur_prs = []
        # get all data with current class
        for i, gt in enumerate(gts):
            pr = prs[i]
            buf_gts = []
            buf_prs = []
            for gtr in gt:
                if gtr['class'] == cl:
                    buf_gts.append(gtr)
            for pred in pr:
                if pred['class'] == cl:
                    buf_prs.append(pred)
            cur_gts.append(buf_gts)
            cur_prs.append(buf_prs)
        # calc ap
        aps.append(AP(cur_gts, cur_prs, threshold, mode))
    return np.mean(aps)


def MOTA(gts, prs, threshold=0.3, vis=False):
    '''
    MOTA metric. Example of gts and prs arrays in main function.
    Input:
        gts - array-like, ground truth array of dicts with keys: 'id', 'bb'
        prs - array-like, predicted array of dicts with keys: 'id', 'bb',
        'conf', 'iou'
        threshold - float, threshold for IOU - if IOU of predicted bounding box
        greater than threshold, it's considered as match, else as mismatch
        vis - boolean, visualize trajectories for debug
    Output:
        mota - float, MOTA
    '''
    m = 0
    fp = 0
    mme = 0
    gt_len = 0
    ids = {1: 1, 2: 2, 3: 3}
    for i, gt in enumerate(gts):
        pr = prs[i]
        for gtr in gt:
            gtr['iou'] = 0
        for pred in pr:
            pred['iou'] = 0
    for i, gt in enumerate(gts):
        pr = prs[i]
        gt_len += len(gt)
        for gtr in gt:
            max_iou = 0
            pred_ind = None
            for pred in pr:
                buf_iou = iou(gtr['bb'], pred['bb'])
                
                if buf_iou > pred['iou'] and buf_iou > max_iou and buf_iou > threshold:
                    pred_ind = pr.index(pred)
                    max_iou = buf_iou
            if pred_ind is not None:
                pr[pred_ind]['iou'] = max_iou
                if pr[pred_ind]['id'] != gtr['id'] and ids[gtr['id']] != pr[pred_ind]['id']:
                    mme += 1
                    ids[gtr['id']] = pr[pred_ind]['id']
                    ids[pr[pred_ind]['id']] = gtr['id']
                pred_ind = None
            else:
                m += 1
                fp += 1
    if vis:
        colors_gt = {}
        colors_pr = {}
        img = np.zeros((200, 200, 3), np.uint8)
        for i, gt in enumerate(gts):
            pr = prs[i]
            for gtr in gt:
                if gtr['id'] not in colors_gt:
                    buf = np.random.randint(0, 255, 3, dtype=np.uint8)
                    colors_gt[gtr['id']] = (int(buf[0]), int(buf[1]), int(buf[2]))#np.random.randint(0, 255, 3)
                col = colors_gt[gtr['id']]
                cv2.rectangle(img, (gtr['bb'][0][0], gtr['bb'][1][1]),
                              (gtr['bb'][1][0], gtr['bb'][0][1]), col)
            for pred in pr:
                if pred['id'] not in colors_pr:
                    buf = np.random.randint(0, 255, 3, dtype=np.uint8)
                    colors_pr[pred['id']] = (int(buf[0]), int(buf[1]), int(buf[2]))
                col = colors_pr[pred['id']]
                cv2.rectangle(img, (pred['bb'][0][0], pred['bb'][1][1]),
                              (pred['bb'][1][0], pred['bb'][0][1]), col)
        cv2.imshow('Tracking', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    # print(m, fp, mme)
    # print(1 - np.sum(m + fp + mme) / np.sum(gt_len))
    return 1 - (m + fp + mme) / gt_len


def MOTP(gts, prs, threshold=100, mode='dist'):
    '''
    MOTP metric. Example of gts and prs arrays in main function, similar for
    MOTA.
    Input:
        gts - array-like, ground truth array of dicts with keys: 'id', 'bb'
        prs - array-like, predicted array of dicts with keys: 'id', 'bb',
        'conf', 'iou'
        threshold - float, threshold for IOU / euclidean distance
        mode - str, 'iou' or 'dist'. If 'iou', calc MOTP using IOU as metric, d
        calculated as 1-IOU. If 'dist', calc MOTP using euclidean distance as
        metric, d calculated as euclidean distance between bounding box centers
    Output:
        motp - float, MOTP
    '''
    d = 0
    c = 0
    if mode == 'iou':
        start_value = 0
    elif mode == 'dist':
        start_value = 1e8
    for i, gt in enumerate(gts):
        pr = prs[i]
        for gtr in gt:
            gtr['iou'] = start_value
        for pred in pr:
            pred['iou'] = start_value
    for i, gt in enumerate(gts):
        pr = prs[i]
        for gtr in gt:
            if mode == 'iou':
                max_iou = 0
            elif mode == 'dist':
                max_iou = 1e8
            pred_ind = None
            for pred in pr:
                if mode == 'iou':
                    buf_iou = iou(gtr['bb'], pred['bb'])
                    if buf_iou > pred['iou'] and buf_iou > max_iou and buf_iou > threshold:
                        pred_ind = pr.index(pred)
                        max_iou = buf_iou
                elif mode == 'dist':
                    c1 = np.array([gtr['bb'][1][0] - gtr['bb'][0][0],
                                   gtr['bb'][1][1] - gtr['bb'][0][1]])
                    c2 = np.array([pred['bb'][1][0] - pred['bb'][0][0],
                                   pred['bb'][1][1] - pred['bb'][0][1]])
                    buf_iou = np.linalg.norm(c1 - c2)
                    if buf_iou < pred['iou'] and buf_iou < max_iou and buf_iou < threshold:
                        pred_ind = pr.index(pred)
                        max_iou = buf_iou
            if pred_ind is not None:
                pr[pred_ind]['iou'] = max_iou
                c += 1
                if mode == 'iou':
                    d += 1 - max_iou
                elif mode == 'dist':
                    d += max_iou
                pred_ind = None
    if c == 0:
        return 0
    else:
        return d / c


def readfiles_for_AP(gt_dir, pr_dir):
    '''
    Function for testing AP and mAP metric. Reads data from test files.
    Input:
        gt_dir - str, path to directory with ground truth files
        pr_dir - str, path to directory with predicted files
    Output:
        gts, prs - ground truth and predcited array of dicts
    '''
    gt_files = sorted(os.listdir(gt_dir))
    gts = []
    for file in gt_files:
        with open(gt_dir + file, 'r') as f:
            lines = f.readlines()
            buf = []
            for line in lines:
                cl = line.split()[0]
                buf_bb = list(map(int, line.split()[1:]))
                bb = [(buf_bb[0], buf_bb[1] + buf_bb[3]),
                      (buf_bb[0] + buf_bb[2], buf_bb[1])]
                buf.append({'class': cl, 'bb': bb})
        gts.append(buf)
    pr_files = sorted(os.listdir(pr_dir))
    prs = []
    for file in pr_files:
        with open(pr_dir + file, 'r') as f:
            lines = f.readlines()
            buf = []
            for line in lines:
                cl = line.split()[0]
                conf = float(line.split()[1])
                buf_bb = list(map(int, line.split()[2:]))
                bb = [(buf_bb[0], buf_bb[1] + buf_bb[3]),
                      (buf_bb[0] + buf_bb[2], buf_bb[1])]
                buf.append({'class': cl, 'conf': conf, 'bb': bb, 'iou': 0})
        prs.append(buf)
    return gts, prs


def main():
    # reading data for accuracy, precision, recall and f1
    fname = 'classification_results.txt'
    with open(fname, 'r') as f:
        lines = f.readlines()
        y_pred = []
        y_true = []
        for line in lines:
            a, b = list(map(int, line.split()))
            y_pred.append(b)
            y_true.append(a)
    print('Sklearn accuracy', metrics.accuracy_score(y_true, y_pred))
    print('Custom accuracy', accuracy(y_true, y_pred))
    print('Sklearn precision', metrics.precision_score(y_true, y_pred,
                                                       average=None))
    print('Custom precision', precision(y_true, y_pred))
    print('Sklearn recall', metrics.recall_score(y_true, y_pred, average=None))
    print('Custom recall', recall(y_true, y_pred))
    print('Sklearn f1', metrics.f1_score(y_true, y_pred, average=None))
    print('Custom f1', f1(y_true, y_pred))
    print('Sklearn RMSE', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print('Custom RMSE', rmse(y_true, y_pred))
    print('Sklearn MAE', metrics.mean_absolute_error(y_true, y_pred))
    print('Custom MAE', mae(y_true, y_pred))
    # reading data for AP and mAP
    gt_dir = './groundtruths/'
    pr_dir = './detections/'
    # data and its format taken from
    # https://github.com/rafaelpadilla/Object-Detection-Metrics
    gts, prs = readfiles_for_AP(gt_dir, pr_dir)
    print('AP', AP(gts, prs))
    # data for MOTA and MOTP
    preds = [[{'id': 1, 'conf': 0.67, 'bb': [(30, 60), (64, 32)], 'iou': 0},
              {'id': 2, 'conf': 0.34, 'bb': [(46, 92), (66, 74)], 'iou': 0},
              {'id': 3, 'conf': 0.56, 'bb': [(10, 160), (30, 150)], 'iou': 0}],
             [{'id': 1, 'conf': 0.78, 'bb': [(54, 86), (73, 64)], 'iou': 0},
              {'id': 2, 'conf': 0.64, 'bb': [(58, 62), (83, 24)], 'iou': 0},
              {'id': 3, 'conf': 0.31, 'bb': [(40, 190), (60, 170)], 'iou': 0}],
             [{'id': 1, 'conf': 0.71, 'bb': [(85, 97), (105, 74)], 'iou': 0},
              {'id': 2, 'conf': 0.82, 'bb': [(98, 60), (130, 30)], 'iou': 0},
              {'id': 3, 'conf': 0.58, 'bb': [(80, 180), (100, 160)], 'iou': 0}]]
    real = [[{'id': 1, 'bb': [(36, 64), (66, 34)]},
             {'id': 2, 'bb': [(48, 89), (68, 69)]},
             {'id': 3, 'bb': [(12, 140), (34, 100)]}],
            [{'id': 1, 'bb': [(56, 64), (86, 34)]},
             {'id': 2, 'bb': [(56, 80), (76, 60)]},
             {'id': 3, 'bb': [(42, 160), (64, 120)]}],
            [{'id': 1, 'bb': [(96, 64), (126, 34)]},
             {'id': 2, 'bb': [(88, 91), (108, 71)]},
             {'id': 3, 'bb': [(79, 150), (99, 110)]}]]
    print('Custom MOTA', MOTA(real, preds))
    print('Custom MOTP', MOTP(real, preds))


if __name__ == '__main__':
    main()
