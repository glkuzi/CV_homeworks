#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:11:07 2020

@author: dell
"""

import numpy as np
import cv2
import re
from tqdm import tqdm


def get_cam_param(filename):
    '''Function for getting camera parameters. Parse calib.txt file.
    Input:
        filename - str, path to calib.txt
    Output:
        K1 - np.array, intrinsic matrix of cam0
        K2 - np.array, intrinsic matrix of cam1
        baseline - float, camera baseline in mm
        doffs - float, x-differnce of principal points
        f - float, focal length in pix
    '''
    with open(filename) as f:
        lines = f.readlines()
    float_extractor = re.compile(r"[-+]?\d*\.\d+|\d+")
    K1 = list(map(float, float_extractor.findall(lines[0].split('=')[-1])))
    K2 = list(map(float, float_extractor.findall(lines[1].split('=')[-1])))
    baseline = float(float_extractor.findall(lines[2].split('=')[-1])[0])
    doffs = float(float_extractor.findall(lines[3].split('=')[-1])[0])
    f = K1[0]
    return np.array(K1).reshape(3, 3), np.array(K2).reshape(3, 3), baseline, doffs, f


def get_p_matrices(K1, K2, doffs):
    '''Calculate camera matrices P.
    Input:
        K1 - np.array, intrinsic matrix of cam0
        K2 - np.array, intrinsic matrix of cam1
        doffs - float, x-differnce of principal points
    Output:
        P1 - np.array, extrinsic matrix of cam0
        P2 - np.array, extrinsic matrix of cam1
        F - np.array, fundamental matrix
    '''
    m1 = np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))
    m2 = np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))
    m2[0, 3] = doffs
    R = np.eye(3)
    t = np.array([doffs, 0, 0]).reshape(-1, 1)
    P1 = np.dot(K1, m1)
    P2 = np.dot(K2, m2)
    # calc fundamental matrix
    k2_buf = np.dot(np.transpose(np.linalg.pinv(K2)), R)
    left_part = np.dot(k2_buf, np.transpose(K1))
    x_buf = np.dot(np.dot(K1, np.transpose(R)), t)
    x_matr = np.array([[0, -x_buf[2], x_buf[1]],
                       [x_buf[2], 0, -x_buf[0]],
                       [-x_buf[1], x_buf[0], 0]])
    F = np.dot(left_part, x_matr)
    return P1, P2, F


def find_closest_point(img1, img2, x, y, window_size=3, thr=None):
    '''Find most 'similar' to (x, y) point on img2.
    Input:
        img1 - np.array, first image
        img2 - np.array, second image
        x - int, x coordinate on img1
        y - int, y coordinate on img1
        window_size - int, window size
    Output:
        x2 - int, x coordinate on img2
        y2 - int, y coordinate on img2
    '''
    wsize = window_size // 2
    x_lr = np.clip([x - wsize, x + wsize+1], 0, img1.shape[1])
    '''
    y_lr = np.clip([y - wsize, y + wsize+1], 0, img1.shape[0])
    buf0 = img1[y_lr[0]:y_lr[1], x_lr[0]:x_lr[1]]
    curr_wind = np.pad(buf0, ((window_size-buf0.shape[0], 0), (window_size-buf0.shape[1], 0), (0, 0)), 'edge')
    '''
    buf0 = img1[y][x_lr[0]:x_lr[1]]
    curr_wind = np.pad(buf0, ((window_size-len(buf0), 0), (0, 0)), 'edge')
    curr_wind = np.array(curr_wind, dtype=np.int16)
    corrs = []
    for i in range(img2.shape[1]):
        x_lr = np.clip([i - wsize, i + wsize+1], 0, img2.shape[1])
        '''
        y_lr = np.clip([y - wsize, y + wsize+1], 0, img2.shape[0])
        buf = img2[y_lr[0]:y_lr[1], x_lr[0]:x_lr[1]]
        wind = np.pad(buf, ((window_size-buf.shape[0], 0), (window_size-buf.shape[1], 0), (0, 0)), 'edge')
        '''
        buf = img2[y][x_lr[0]:x_lr[1]]
        wind = np.pad(buf, ((window_size-len(buf), 0), (0, 0)), 'edge')
        wind = np.array(wind, dtype=np.int16)
        corrs.append(np.sum(np.abs((curr_wind - wind))))
        if thr is not None and corrs[-1] < thr:
            break
    y2 = y
    x2 = np.argmin(corrs)
    return x2, y2


def disp_to_z(disp_map, baseline, f, doffs):
    '''Transform disparity map into depth map.
    Input:
        disp_map - np.array, disparity map
        baseline - float, camera baseline in mm
        doffs - float, x-differnce of principal points
        f - float, focal length in pix
    Output:
        depth_map - np.array, depth map
    '''
    depth_map = np.zeros(disp_map.shape, dtype=np.float32)
    for i in range(disp_map.shape[0]):
        for j in range(disp_map.shape[1]):
            depth_map[i][j] = baseline * f / (disp_map[i][j] + doffs)
    return depth_map


def main():
    folder = './Adirondack-perfect'
    calib_file = folder + '/calib.txt'
    K1, K2, baseline, doffs, f = get_cam_param(calib_file)
    get_p_matrices(K1, K2, doffs)
    print('Camera parameters:', get_cam_param(calib_file))
    path1 = folder + '/im0.png'
    path2 = folder + '/im1.png'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    sf = 4
    # img1 = img1[970:1120, 1260:1400]
    # img2 = img2[970:1120, 1094:1234]
    img1 = cv2.resize(img1, (img1.shape[1]//sf, img1.shape[0]//sf))
    img2 = cv2.resize(img2, (img2.shape[1]//sf, img2.shape[0]//sf))
    cv2.imshow('im1', img1)
    cv2.imshow('im2', img2)

    disp_map = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.int32)
    for x1 in tqdm(range(img1.shape[1])):
        for y1 in range(img1.shape[0]):
            x2, y2 = find_closest_point(img1, img2, x1, y1)
            disp_map[y1][x1] = x2-x1

    n_map = np.array(disp_map + np.min(disp_map), dtype=np.uint8)
    dmap = disp_to_z(disp_map, baseline, f, doffs)
    ndmap = np.array(dmap, dtype=np.uint16)
    cv2.imwrite('dmap_0_w5_big.pgm', ndmap)
    cv2.imwrite('disp_0_w5_big.pgm', n_map)

    disp_map = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.int32)
    for x1 in tqdm(range(img2.shape[1])):
        for y1 in range(img2.shape[0]):
            x2, y2 = find_closest_point(img2, img1, x1, y1)
            disp_map[y1][x1] = x2-x1
    # print(np.max(disp_map), np.min(disp_map))
    n_map = np.array(disp_map + np.min(disp_map), dtype=np.uint8)
    dmap = disp_to_z(disp_map, baseline, f, doffs)
    ndmap = np.array(dmap, dtype=np.uint16)
    cv2.imwrite('dmap_1_w5_big.pgm', ndmap)
    cv2.imwrite('disp_1_w5_big.pgm', n_map)
    # print(np.max(dmap), np.min(dmap))
    cv2.imshow('disp', n_map)
    cv2.imshow('dmap', ndmap)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
