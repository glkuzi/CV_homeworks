#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from collections import OrderedDict


def check_4_circle_points(buf_img, i, j, threshold, radius=3):
    '''Function for checking only 1, 5, 9, 13 pixels.
    Input:
        buf_img - np.array, input image in grayscale
        i - int, index of pixel
        j - int, index of pixel
        threshold - int, threshold value for FAST
    Output:
        bool, True if 3 from 4 pixels are darker or brighter, else False
    '''
    points = [buf_img[np.clip(i-radius, 0, buf_img.shape[0]-1)][j],
              buf_img[np.clip(i+radius, 0, buf_img.shape[0]-1)][j],
              buf_img[i][np.clip(j-radius, 0, buf_img.shape[1]-1)],
              buf_img[i][np.clip(j+radius, 0, buf_img.shape[1]-1)]]
    darker = 0
    similar = 0
    brighter = 0
    for point in points:
        if point < buf_img[i][j] - threshold:
            darker += 1
        elif point > buf_img[i][j] + threshold:
            brighter += 1
        else:
            similar += 1
    if darker >= 3 or brighter >= 3:
        return True
    else:
        return False


def check_n_points(buf_img, i, j, threshold, n, radius=3):
    '''Function for checking 16 pixels.
    Input:
        buf_img - np.array, input image in grayscale
        i - int, index of pixel
        j - int, index of pixel
        threshold - int, threshold value for FAST
        n - int, number of pixels for special point detection
    Output:
        bool, True if n pixels are darker or brighter, else False
    '''
    darker = 0
    similar = 0
    brighter = 0
    for k in range(i - radius, i + radius + 1):
        y_off = np.sqrt(radius ** 2 - (k - i) ** 2)
        y1 = np.clip(np.int32(np.round(j - y_off)), 0, buf_img.shape[1]-1)
        y2 = np.clip(np.int32(np.round(j + y_off)), 0, buf_img.shape[1]-1)
        k_clip = np.clip(k, 0, buf_img.shape[0]-1)
        point = buf_img[k_clip][y1]
        if point < buf_img[i][j] - threshold:
            darker += 1
        elif point > buf_img[i][j] + threshold:
            brighter += 1
        else:
            similar += 1
        point = buf_img[k_clip][y2]
        if point < buf_img[i][j] - threshold:
            darker += 1
        elif point > buf_img[i][j] + threshold:
            brighter += 1
        else:
            similar += 1
        if darker >= n or brighter >= n:
            return True
    if darker >= n or brighter >= n:
        return True
    else:
        return False


def fast(img, threshold=20, n=12):
    ''' FAST feature detector.
    Input:
        img - np.array, input image
        threshold - int, FAST threshold
        n - int, number of points for special point
    Output:
        special_points - list, each element - tuple with indices of special
        point
    '''
    if img.shape[-1] == 3:
        # to grayscale
        buf_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        buf_img = img.copy()
    special_points = []
    for i in range(3, buf_img.shape[0]-3):
        for j in range(3, buf_img.shape[1]-3):
            # get circle
            if check_4_circle_points(buf_img, i, j, threshold):
                if check_n_points(buf_img, i, j, threshold, n):
                    special_points.append((i, j))
    return special_points


def get_top_n_points(img, special_points, window_size=3, k=0.05, top_n=1000):
    '''Function for sorting special points, generated by FAST. Uses Harris
    measure as metric.
    Input:
        img - np.array, input image
        special_points - list, each element - tuple with indices of special
        point
        window_size - int, size of window for calculating metric
        k - float, metric parameter
        top_n - number of points with better metric to be returned
    Output:
        measures - dict, keys - tuple with indices of special point, values -
        metric value for special point
    '''
    block_size = window_size
    k_prime = k / (1 - k)
    aperture_size = 3
    if img.shape[-1] == 3:
        # to grayscale
        buf_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        buf_img = img.copy()
    harr_matr = cv2.cornerEigenValsAndVecs(buf_img, block_size, aperture_size)
    measures = {}
    for point in special_points:
        l1, l2, x1, y1, x2, y2 = harr_matr[point[0]][point[1]]
        measures[point] = l1 * l2 - k * (l1 + l2) ** 2
        # another version of measure
        # measures[point] = (l1 - k_prime * l2) * (l2 - k_prime * l1)
    measures = {k: v for k, v in sorted(measures.items(),
                                        key=lambda item: item[1])}
    measures = list(measures.keys())[-top_n:]
    return measures


def pyramid_fast(img, threshold=20, n=9, window_size=3, k=0.05, top_n=500,
                 pyr_depth=3, scale=1.2):
    '''Function for generation of multiscale FAST features.
    Input:
        img - np.array, input image
        threshold - int, FAST threshold
        n - int, number of points for special point
        window_size - int, size of window for calculating metric
        k - float, metric parameter
        top_n - number of points with better metric to be returned
        pyr_depth - int, number of images in pyramid
    Output:
        top_n_points - dict, keys - tuple with indices of special point, values
        - metric value for special point
    '''
    buf_points = fast(img, threshold, n)
    spec_points = get_top_n_points(img, buf_points, window_size,
                                   k, len(buf_points))
    downscaled_img = img.copy()
    for depth in range(pyr_depth):
        result = []
        w = int(downscaled_img.shape[1] / scale)
        h = int(downscaled_img.shape[0] / scale)
        downscaled_img = cv2.resize(downscaled_img, (w, h))
        # downscaled_img = cv2.pyrDown(downscaled_img)
        buf_points = fast(downscaled_img, threshold, n)
        buf_points = get_top_n_points(img, buf_points, window_size, k,
                                      len(buf_points))
        for p in spec_points:
            for p1 in buf_points:
                if p[0] == int(scale ** (depth + 1) * p1[0]) and p[1] == int(scale ** (depth + 1) * p1[1]):
                    result.append(p)
                    break
        spec_points = result.copy()
    top_n_points = get_top_n_points(img, spec_points, window_size, k, top_n)
    return top_n_points


def calc_moment(buf_img, point, p, q):
    '''Function for moment calculation.
    Input:
        buf_img - np.array, grayscale image
        point - tuple, contains coordinates of point on image
        p - int, moment order for x
        q - int, moment order for y
    Output:
        moment - int, calculated moment for point
    '''
    i, j = point
    points = [(i-3, j), (i-3, j-1), (i-3, j+1),
              (i+3, j), (i+3, j-1), (i+3, j+1),
              (i, j-3), (i+1, j-3), (i-1, j-3),
              (i, j+3), (i-1, j+3), (i+1, j+3),
              (i-2, j-2), (i-2, j+2), (i+2, j-2),
              (i+2, j+2)]

    moment = 1 ** p * 1 ** q * buf_img[i][j]
    for po in points:
        moment += (po[0] - i) ** p * (po[1] - j) ** q * buf_img[po[0]][po[1]]
    '''
    moment = 0
    radius = 3
    for k in range(i - radius, i + radius + 1):
        y_off = np.sqrt(radius ** 2 - (k - i) ** 2)
        y1 = np.int32(np.round(j - y_off))
        y2 = np.int32(np.round(j + y_off))
        #print(k, i, y_off, y1, y2)
        #for m in range(j-radius, j+radius+1):
        for m in range(y1, y2+1):
            moment += k ** p * m ** q * buf_img[k][m]
    '''
    return moment


def calc_centroid(img, special_points):
    '''Function for intencity centroid calculation.
    Input:
        img - np.array, input image
        special_points - list, each element - tuple with indices of special
        point
    Output:
        points_dict - dict, key - tuple with indices of special point, value -
        intencity centroid of this point
    '''
    if img.shape[-1] == 3:
        # to grayscale
        buf_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        buf_img = img.copy()
    points_dict = {}
    for point in special_points:
        m10 = calc_moment(buf_img, point, 1, 0)
        m01 = calc_moment(buf_img, point, 0, 1)
        # m00 = calc_moment(buf_img, point, 0, 0, radius)
        points_dict[point] = np.arctan2(m01, m10)
    return points_dict


def compute_points(window_shape, size):
    '''Function for computing fixed points for BRIEF.
    Input:
        window_shape - tuple, shape of window for descriptor
        size - int, number of points for descriptor
    Output:
        points_dict - dict, key - angles from -pi to pi with step pi / 15,
        values - numpy array with points, rotated on this angle
    '''
    byte = size
    np.random.seed(200)
    angles = np.arange(-np.pi, np.pi + 0.01, np.pi / 15)
    # generate coordinates with fixed seed
    x_coords = np.clip(np.random.normal(window_shape[0] / 2,
                                        window_shape[0] / 6, 2 * byte),
                       0, window_shape[0]-1) - window_shape[0] / 2
    y_coords = np.clip(np.random.normal(window_shape[1] / 2,
                                        window_shape[1] / 6, 2 * byte),
                       0, window_shape[1]-1) - window_shape[1] / 2
    points = np.array([list(x_coords), list(y_coords)])
    points_dict = {}
    for ang in angles:
        # for all angles rotate coordinates and save its to the dict
        rotation = np.array([[np.cos(ang), -np.sin(ang)],
                             [np.sin(ang), np.cos(ang)]])
        points_dict[ang] = np.dot(rotation,
                                  points) + np.array([[window_shape[0] / 2],
                                                      [window_shape[1] / 2]])
    return points_dict


def steered_brief(img, points_dict, size=32,
                  blur_size=5, window_size=31):
    '''Function for computing steered BRIEF descriptors.
    Input:
        img - np.array, input image
        points_dict - dict, key - angles from -pi to pi with step pi / 15,
        values - numpy array with points, rotated on this angle
        size - int, size of descriptor
        blur_size - int, size of window for bulring image
        window_size - int, size of window from that choosed points for
        descriptor
    Output:
        descriptors - dict, key - tuple with indices of special point, values -
        descriptors of this point
    '''
    if img.shape[-1] == 3:
        # to grayscale
        buf_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        buf_img = img.copy()
    byte = 8
    blurred_img = cv2.GaussianBlur(buf_img, (blur_size, blur_size), 0)
    descriptors = OrderedDict()
    for point in list(points_dict.keys()):
        # get window
        left = max(0, point[0] - ((window_size - 1) // 2))
        right = min(point[0] + ((window_size - 1) // 2), blurred_img.shape[0])
        down = max(0, point[1] - ((window_size - 1) // 2))
        up = min(point[1] + ((window_size - 1) // 2), blurred_img.shape[1])
        window = blurred_img[left:right+1, down:up+1]
        point_descr = []
        theta = points_dict[point]
        points_by_angle = compute_points((window_size,
                                          window_size), byte * size)
        ind = np.argmin(np.abs(np.array(list(points_by_angle.keys())) - theta))
        key = list(points_by_angle.keys())[ind]
        points_list = points_by_angle[key]
        for k in range(size):
            coords = points_list[:, k: (k+1) * size]
            descr = 0
            for b in range(byte):
                x1 = np.clip(int(coords[0][2 * b]), 0, window.shape[0] - 1)
                x2 = np.clip(int(coords[0][2 * b + 1]), 0, window.shape[0] - 1)
                y1 = np.clip(int(coords[1][2 * b]), 0, window.shape[1] - 1)
                y2 = np.clip(int(coords[1][2 * b + 1]), 0, window.shape[1] - 1)
                if window[x1][y1] < window[x2][y2]:
                    descr += 2 ** b
            point_descr.append(descr)
        descriptors[point] = point_descr
    return descriptors


def orb(img, fast_threshold=20, n=9, window_size=3, k=0.05, top_n=1000,
        pyr_depth=3, scale=1.2, size=128, blur_size=5, brief_size=31):
    '''ORB.
    Input:
        img - np.array, input image
        fast_threshold - int, FAST threshold
        n - int, number of points for special point
        window_size - int, size of window for calculating metric
        k - float, metric parameter
        top_n - number of points with better metric to be returned
        pyr_depth - int, number of images in pyramid
        scale - float, scale factor of pyramid
        size - int, size of descriptor
        blur_size - int, size of window for bulring image
        brief_size - int, size of window from that choosed points for
        descriptor
    Output:
        kps - list of cv2.Keypoint, contain special points
        dsc - list, contains descriptors for special points
    '''
    top_ns = pyramid_fast(img, fast_threshold, n, window_size, k, top_n,
                          pyr_depth, scale)
    points_dict = calc_centroid(img, top_ns)
    descs = steered_brief(img, points_dict, size, blur_size, brief_size)
    kps = [cv2.KeyPoint(y, x, brief_size) for (x, y) in list(descs.keys())]
    dsc = list(descs.values())
    return kps, dsc


# block with functions with image transformations for ORB testing
def affine(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))


def perspective(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (300, 300))


def rescale(img):
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)


def rotate(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def main():
    img = cv2.imread('./dragonfly.jpg')
    resc_img = rescale(img)
    rot_img = rotate(img, 33)
    perp_img = perspective(img)
    af_img = affine(img)
    test_images = [resc_img, rot_img, perp_img, af_img]
    st = time.time()
    orbcv = cv2.ORB_create()
    ks1, ds1 = orbcv.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                      None)
    kss = []
    dss = []
    for image in test_images:
        ks2, ds2 = orbcv.detectAndCompute(cv2.cvtColor(image,
                                                       cv2.COLOR_BGR2GRAY),
                                          None)
        kss.append(ks2)
        dss.append(ds2)
    end = time.time()
    print('OpenCV orb time:', end - st)
    st = time.time()
    kps1, dsc1 = orb(img)
    for p in kps1:
        cv2.circle(img, (int(p.pt[0]), int(p.pt[1])), 3, (0, 0, 255))
    kpss = []
    dscs = []
    for image in test_images:
        kps2, dsc2 = orb(image)
        for p in kps2:
            cv2.circle(image, (int(p.pt[0]), int(p.pt[1])), 3, (0, 0, 255))
        kpss.append(kps2)
        dscs.append(dsc2)
    end = time.time()
    print('My orb time:', end-st)
    names = ['rescaled', 'rotated', 'perspective', 'affine']
    for i in range(len(test_images)):
        length = min(len(dsc1), len(dscs[i]))
        dsc1 = np.ndarray((length, len(dsc1[0])), buffer=np.array(dsc1[:length]),
                          dtype=np.uint8)
        dsc2 = np.ndarray((length, len(dscs[i][0])), buffer=np.array(dscs[i][:length]),
                          dtype=np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(dsc1, dsc2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img, kps1, test_images[i], kpss[i],
                               matches[:10], None, flags=2)
        cv2.imwrite(names[i] + '.png', img3)

    for i in range(len(test_images)):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches1 = bf.match(ds1, dss[i])
        # Sort them in the order of their distance.
        matches1 = sorted(matches, key=lambda x: x.distance)
        img4 = cv2.drawMatches(img, ks1, test_images[i], kss[i],
                               matches1[:10], None, flags=2)
        cv2.imwrite('cv_' + names[i] + '.png', img4)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
