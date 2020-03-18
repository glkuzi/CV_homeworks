#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import skimage.metrics


def count_disparity(res, cv_res):
    # prevent from transfomation -1 into 255
    res_img = np.abs(np.asarray(res, dtype=np.int32) - np.asarray(cv_res, dtype=np.int32))
    return np.asarray(res_img, dtype=np.uint8)


def count_prim_metric(res, cv_res):
    return np.sum(count_disparity(res, cv_res))


def image_padding(img, pad, mode='edge'):
    pads = [(pad[0], pad[1])] * 2 + [(0, 0)]
    img_pad = np.pad(img, pads, mode)
    return img_pad


def moving_average(img, size=(3, 3), debug=False):
    kernel = np.ones(size)
    pad = (size[0] // 2, size[1] // 2)
    img_pad = image_padding(img, pad)
    for i in range(size[0]//2, img_pad.shape[0] - size[0]//2):
        for j in range(size[1]//2, img_pad.shape[1] - size[1]//2):
            for k in range(3):
                if i == 1 and j == 1 and debug:
                    print(img_pad[i-size[0]//2:i+size[0]//2+1,
                                  j-size[1]//2:j+size[1]//2+1, k])
                    print(np.sum(kernel * img_pad[i-size[0]//2:i+size[0]//2+1,
                                                  j-size[1]//2:j+size[1]//2+1,
                                                  k]))
                img_pad[i][j][k] = np.floor(np.sum(kernel * img_pad[i-size[0]//2:i+size[0]//2+1,j-size[1]//2:j+size[1]//2+1, k]) / (size[0] * size[1]))
    return img_pad[size[0]//2:size[0]//2 * (-1), size[1]//2:size[1]//2 * (-1)]


def main():
    # open image
    size = (3, 3)
    img = cv2.imread('./Brocolli.jpg')

    res = moving_average(img, size)
    print(res.shape, img.shape)
    cv2.imshow('Result', res)
    # kernel = np.ones(size) / (size[0] * size[1])
    # cv_res = cv2.filter2D(img,-1,kernel)
    # print(res[0,0], cv_res[0,0], cv_res[0,0] - res[0,0])
    cv_res = cv2.blur(img, size)
    cv2.imshow('Original', img)
    cv2.imshow('CV_res', cv_res)
    # show delta of images
    delta = count_disparity(res, cv_res)
    cv2.imshow('Delta', delta)
    # print metrics
    print('Metric =', count_prim_metric(res, cv_res))
    print('MSE =', skimage.metrics.mean_squared_error(res, cv_res))
    print('PSNR =', skimage.metrics.peak_signal_noise_ratio(res, cv_res))
    cv2.imwrite('./Brocolli_res.jpg', res)
    cv2.imwrite('./Brocolli_cv_res.jpg', cv_res)
    cv2.imwrite('./Brocolli_delta.jpg', delta)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
