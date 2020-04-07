#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:37:14 2020

@author: dell
"""


import numpy as np
import cv2


def rotate(img, angle):
    '''Rotate 2d image.
    Input:
        img - np.array, opencv image to be rotated
        angle - int, angle of rotation in degrees
    Output:
        rot_img - np.array, rotated image
    '''
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
    '''Resize 2d image.
    Input:
        img - np.array, opencv image to be resized
        fx - float, x coeff
        fy - float, y coeff
    Output:
        rot_img - np.array, resized image
    '''
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
    # instead of interpolation
    if fx > 1.0 or fy > 1.0:
        grayImage = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        (thresh, mask) = cv2.threshold(grayImage, 1, 255, cv2.THRESH_BINARY_INV)
        rot_img = cv2.inpaint(rot_img, mask, 3, cv2.INPAINT_TELEA)
    return rot_img


def translation(img, tx, ty):
    '''Translation transform for 2d image.
    Input:
        img - np.array, opencv image
        tx - int, translation coefficient for x-axis in pixels
        ty - int, translation coefficient for y-axis in pixels
    Output:
        rot_img - np.array, transformed image
    '''
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
    '''Shear transform for 2d image.
    Input:
        img - np.array, opencv image
        sx - float, x coeff
        sy - float, y coeff
    Output:
        rot_img - np.array, transformed image
    '''
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


def rotate3d(img, angle, axis):
    '''Rotation transform for 3d image. Work with .ppm file format.
    Input:
        img - np.array, opencv image
        angle - int, rotation angle in degrees
        axis - int, rotation axis; must be 0 for rotation around x, 1 for y,
                    2 for z
    Output:
        rot_img - np.array, rotated image
    '''
    angle = angle / 180 * np.pi
    a = np.cos(angle)
    b = np.sin(angle)
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        raise ValueError('Wrong file format')
    (cX, cY, cZ) = (w // 2, h // 2, np.max(img) // 2)
    if axis == 0:
        rot_matr = np.array([[1, 0, 0, 0],
                             [0, a, b * cY / cZ, (1 - a) * cY - b * cY],
                             [0, -b * cZ / cY, a, b * cZ + (1 - a) * cZ]])
    elif axis == 1:
        rot_matr = np.array([[a, 0, b * cX / cZ, (1 - a) * cX - b * cX],
                             [0, 1, 0, 0],
                             [-b * cZ / cX, 0, a, b * cZ + (1 - a) * cZ]])
    elif axis == 2:
        rot_matr = np.array([[a, b, 0, (1 - a) * cX - b * cY],
                             [-b, a, 0, b * cX + (1 - a) * cY],
                             [0, 0, 1, 0]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1, z1 = rot_matr @ np.array([float(x), float(y), float(img[y, x]), 1.0]).T
            xs = np.clip([int(np.floor(x1)), int(np.round(x1)), int(np.ceil(x1))], 0, w - 1)
            ys = np.clip([int(np.floor(y1)), int(np.round(y1)), int(np.ceil(y1))], 0, h - 1)
            zs = np.clip(int(z1), 0, np.iinfo(img[0, 0]).max)
            for i in range(3):
                rot_img[ys[i], xs[i]] = zs
    return rot_img


def resize3d(img, fx, fy, fz):
    '''Resize transform for 3d image. Work with .ppm file format.
    Input:
        img - np.array, opencv image
        fx - float, resize coefficient for x-axis
        fy - float, resize coefficient for y-axis
        fz - float, resize coefficient for z-axis
    Output:
        rot_img - np.array, resized image
    '''
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((int(fy * h), int(fx * w)), np.uint16)
    else:
        raise ValueError('Wrong file format')
    rot_matr = np.array([[fx, 0, 0],
                         [0, fy, 0],
                         [0, 0, fz]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1, z1 = rot_matr @ np.array([float(x), float(y), float(img[y, x])]).T
            xs = np.clip(int(np.round(x1)), 0, int(fx * w) - 1)
            ys = np.clip(int(np.round(y1)), 0, int(fy * h) - 1)
            zs = np.clip(int(z1), 0, np.iinfo(img[0, 0]).max)
            rot_img[ys, xs] = zs
    if fx > 1.0 or fy > 1.0:
        grayImage = np.array(rot_img, dtype=np.uint8)
        (thresh, mask) = cv2.threshold(grayImage, 1, 255, cv2.THRESH_BINARY_INV)
        rot_img = cv2.inpaint(rot_img, mask, 3, cv2.INPAINT_TELEA)
    return rot_img


def translation3d(img, tx, ty, tz):
    '''Resize transform for 3d image. Work with .ppm file format.
    Input:
        img - np.array, opencv image
        tx - int, translation coefficient for x-axis in pixels
        ty - int, translation coefficient for y-axis in pixels
        tz - int, translation coefficient for z-axis in pixels
    Output:
        rot_img - np.array, transformed image
    '''
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        raise ValueError('Wrong file format')
    tr_matr = np.array([[1, 0, 0, tx],
                        [0, 1, 0, ty],
                        [0, 0, 1, tz]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1, z1 = tr_matr @ np.array([float(x), float(y), float(img[y, x]), 1.0]).T
            xs = np.clip(int(np.round(x1)), 0, w - 1)
            ys = np.clip(int(np.round(y1)), 0, h - 1)
            zs = np.clip(int(z1), 0, np.iinfo(img[0, 0]).max)
            rot_img[ys, xs] = zs
    return rot_img


def shear3d(img, sxy=0.0, sxz=0.0, syx=0.0, syz=0.0, szx=0.0, szy=0.0):
    '''Shear transform for 3d image. Work with .ppm file format.
    Input:
        img - np.array, opencv image
        sxy, sxz, syx, syz, szx, szy - float, shear coefficients
    Output:
        rot_img - np.array, transformed image
    '''
    (h, w) = img.shape[:2]
    d = 1 if len(img.shape) == 2 else img.shape[2]
    if d == 1:
        # ppm format
        rot_img = np.zeros((h, w), np.uint16)
    else:
        raise ValueError('Wrong file format')
    (cX, cY, cZ) = (w // 2, h // 2, np.max(img) // 2)
    s_matr = np.array([[1.0, syx * cX / cY, szx * cX / cZ, -(syx + szx) * cX],
                       [sxy * cY / cX, 1.0, szy * cY / cZ, -(sxy + szy) * cY],
                       [sxz * cZ / cX, syz * cZ / cY, 1.0, -(sxz + syz) * cZ]])
    coords_x = np.arange(0, w)
    coords_y = np.arange(0, h)
    for x in coords_x:
        for y in coords_y:
            x1, y1, z1 = s_matr @ np.array([float(x), float(y), float(img[y, x]), 1.0]).T
            xs = np.clip([int(np.floor(x1)), int(np.round(x1)), int(np.ceil(x1))], 0, w - 1)
            ys = np.clip([int(np.floor(y1)), int(np.round(y1)), int(np.ceil(y1))], 0, h - 1)
            zs = np.clip(int(z1), 0, np.iinfo(img[0, 0]).max)
            for i in range(3):
                rot_img[ys[i], xs[i]] = zs
    return rot_img


def main():
    img_path = './dragonfly.jpg'
    path_3d = './image-depth/d-1316653648.611579-1109571627.pgm'
    # path_3d = './image-depth/r-1316653580.484909-1316500621.ppm'
    img = cv2.imread(img_path)
    img3d = cv2.imread(path_3d, cv2.IMREAD_UNCHANGED)
    rot_img = rotate(img, 45)
    cv2.imwrite('rot.png', rot_img)
    rot_img = resize(img, 0.5, 0.5)
    cv2.imwrite('res05.png', rot_img)
    rot_img = resize(img, 1.5, 1.5)
    cv2.imwrite('res15.png', rot_img)
    rot_img = translation(img, 50, 40)
    cv2.imwrite('tr5040.png', rot_img)
    rot_img = shear(img, 0.2, 0.3)
    cv2.imwrite('sh0203.png', rot_img)

    rot_img = rotate3d(img3d, 45, 0)
    cv2.imwrite('rot3d0.pgm', rot_img)
    rot_img = rotate3d(img3d, 45, 1)
    cv2.imwrite('rot3d1.pgm', rot_img)
    rot_img = rotate3d(img3d, 45, 2)
    cv2.imwrite('rot3d2.pgm', rot_img)
    rot_img = resize3d(img3d, 0.5, 0.5, 0.5)
    cv2.imwrite('res3d05.pgm', rot_img)
    rot_img = resize3d(img3d, 1.5, 1.5, 1.5)
    cv2.imwrite('res3d15.pgm', rot_img)
    rot_img = translation3d(img3d, 50, 40, 10000)
    cv2.imwrite('tr3d504010000.pgm', rot_img)
    rot_img = shear3d(img3d, 0.2, 0.3, 0.2)
    cv2.imwrite('sh3d020302.pgm', rot_img)


if __name__ == '__main__':
    main()
