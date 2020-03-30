#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:37:14 2020

@author: dell
"""


import numpy as np
import cv2


def rotation(img, angle):
    angle = angle / 180 * np.pi
    a = np.cos(angle)
    b = np.sin(angle)
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        rot_img = np.zeros((h, w, d), np.uint8)
    (cX, cY) = (w // 2, h // 2)
    rot_matr = np.array([[a, b, (1 - a) * cX - b * cY],
                         [-b, a, b * cX + (1 - a) * cY]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1 = rot_matr @ np.array([float(x), float(y), 1.0]).T
            xs = np.clip([int(np.floor(x1)), int(np.round(x1)), int(np.ceil(x1))], 0, w - 1)
            ys = np.clip([int(np.floor(y1)), int(np.round(y1)), int(np.ceil(y1))], 0, h - 1)
            for i in range(3):
                rot_img[ys[i], xs[i]] = img[y, x]
    return rot_img


def resize(img, fx, fy):
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((int(fy * h), int(fx * w)), np.uint16)
    else:
        rot_img = np.zeros((int(fy * h), int(fx * w), d), np.uint8)
    rot_matr = np.array([[fx, 0],
                         [0, fy]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1 = rot_matr @ np.array([float(x), float(y)]).T
            xs = np.clip(int(np.round(x1)), 0, int(fx * w) - 1)
            ys = np.clip(int(np.round(y1)), 0, int(fy * h) - 1)
            rot_img[ys, xs] = img[y, x]
    return rot_img


def translation(img, tx, ty):
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        rot_img = np.zeros((h, w, d), np.uint8)
    tr_matr = np.array([[1, 0, tx],
                        [0, 1, ty]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1 = tr_matr @ np.array([float(x), float(y), 1.0]).T
            xs = np.clip(int(np.round(x1)), 0, w - 1)
            ys = np.clip(int(np.round(y1)), 0, h - 1)
            rot_img[ys, xs] = img[y, x]
    return rot_img


def shear(img, sx, sy):
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        rot_img = np.zeros((h, w, d), np.uint8)
    (cX, cY) = (w // 2, h // 2)
    s_matr = np.array([[1.0, sx, -sx * cX],
                        [sy, 1.0, -sy * cY]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1 = s_matr @ np.array([float(x), float(y), 1.0]).T
            xs = np.clip([int(np.floor(x1)), int(np.round(x1)), int(np.ceil(x1))], 0, w - 1)
            ys = np.clip([int(np.floor(y1)), int(np.round(y1)), int(np.ceil(y1))], 0, h - 1)
            for i in range(3):
                rot_img[ys[i], xs[i]] = img[y, x]
    return rot_img


def main():
    img_path = './dragonfly.jpg'
    path_3d = './image-depth/d-1316653648.611579-1109571627.pgm'
    #path_3d = './image-depth/r-1316653580.484909-1316500621.ppm'
    img = cv2.imread(img_path)
    new_img = cv2.imread(path_3d, cv2.IMREAD_UNCHANGED)
    cv2.imshow('3d img', new_img)
    print(new_img.shape)
    print(np.min(new_img), np.max(new_img))
    #rot_img = rotation(new_img, -45)
    #rot_img = resize(new_img, 1.0, 0.5)
    #rot_img = translation(new_img, 20, 30)
    rot_img = shear(img, 0.0, 1.0)
    #cv2.imshow('Start image', rot_img)
    cv2.imshow('rot img', rot_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()