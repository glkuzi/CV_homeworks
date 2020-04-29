#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os


def simple_lucas_kanade(prev_img, next_img, prev_pts, win_size=25, mode='pyr'):
    '''
    Realization of simple Lucas-Kanade method, without image pyramid
    Input:
        prev_img - np.array, grayscale image of previous frame of video
        next_img - np.array, grayscale image of current frame of video
        prev_pts - np.array, special points to be tracked
        win_size - int, window size for Lucas-Kanade method
        mode - str, could be 'pyr' or 'simple'. In 'pyr' mode returns points
        velocity, in 'simple' mode returns new points.
    Output:
        velocities or new_points - np.array, depends on choosen mode
    '''
    # calc derivatives
    der_x, der_y = cv2.spatialGradient(next_img)
    der_t = cv2.subtract(next_img, prev_img)
    half = win_size // 2
    # for every point calc v and u in window
    new_points = np.zeros_like(prev_pts)
    veloc = np.zeros_like(prev_pts)
    for i, point in enumerate(prev_pts):
        x_c, y_c = point[0][0], point[0][1]
        # get coords of window
        # ADDED 1 TO GET REAL WINDOW
        x_coords = np.clip([x_c - half, x_c + half + 1], 0, next_img.shape[1])
        y_coords = np.clip([y_c - half, y_c + half + 1], 0, next_img.shape[0])
        # x1, x2 = int(np.rint(x_coords[0])), int(np.rint(x_coords[1]))
        # y1, y2 = int(np.rint(y_coords[0])), int(np.rint(y_coords[1]))
        x1, x2 = int(x_coords[0]), int(x_coords[1])
        y1, y2 = int(y_coords[0]), int(y_coords[1])
        cur_dx = der_x[y1:y2, x1:x2]
        cur_dy = der_y[y1:y2, x1:x2]
        cur_dt = der_t[y1:y2, x1:x2]
        A = np.array([[np.sum(cur_dx ** 2), np.sum(cur_dx * cur_dy)],
                      [np.sum(cur_dx * cur_dy), np.sum(cur_dy ** 2)]])
        b = np.array([-np.sum(cur_dx * cur_dt), -np.sum(cur_dy * cur_dt)]).T
        v, u = np.linalg.pinv(A) @ b
        new_points[i][0] = [x_c + v, y_c + u]
        veloc[i][0] = [v, u]
    if mode == 'pyr':
        return veloc
    if mode == 'simple':
        return new_points


def get_pyramid(prev_img, next_img, pyr_size=3):
    '''
    Return image pyramid. Pyramid taken with scale factor 2.
    Input:
        prev_img - np.array, grayscale image of previous frame of video
        next_img - np.array, grayscale image of current frame of video
        pyr_size - int, pyramid size
    Output:
        prevs, next - np.array, arrays with image pyramid
    '''
    prevs = [prev_img]
    nexts = [next_img]
    for i in range(pyr_size):
        prevs.append(cv2.pyrDown(prevs[-1]))
        nexts.append(cv2.pyrDown(nexts[-1]))
    return prevs, nexts


def pyr_lucas_kanade(prev_img, next_img, prev_pts, win_size=15, pyr_size=3):
    '''
    Realization of Lucas-Kanade method with image pyramid
    Input:
        prev_img - np.array, grayscale image of previous frame of video
        next_img - np.array, grayscale image of current frame of video
        prev_pts - np.array, special points to be tracked
        win_size - int, window size for Lucas-Kanade method
        pyr_size - int, pyramid size
    Output:
        new_points - np.array, shifted points
    '''
    # loop for all image in pyramid
    prevs, nexts = get_pyramid(prev_img, next_img, pyr_size)
    cur_pts = prev_pts / (2 ** (pyr_size - 1))
    velocs = np.zeros_like(prev_pts)
    for k in range(pyr_size, 0, -1):
        cur_prev = prevs[k - 1]
        cur_next = nexts[k - 1]
        veloc = simple_lucas_kanade(cur_prev, cur_next, cur_pts, win_size)
        if k != 1:
            velocs += veloc * (2 ** (k - 1))
            cur_pts = (prev_pts + veloc * (2 ** (k - 1))) / (2 ** (k - 1)) * 2
    return prev_pts + velocs


def epe(v_pred, v_true, N):
    '''
    Endpoint error function.
    Input:
        v_pred - np.array, predicted special points
        v_true - np.array, true special points
        N - int, number of points. Used instead of image size due to
        Lucas-Kanade method search sparse optical flow.
    Output:
        loss - int, endpoint error loss
    '''
    loss = np.sum(np.sqrt((v_true[:, :, 0] - v_pred[:, :, 0]) ** 2 + (v_true[:, :, 1] - v_pred[:, :, 1]) ** 2)) / N
    return loss


def calc_epe(frames, path, lk_params, custom_lk_params, visualize=False):
    '''
    Function for EPE calculation on SINTEL test part. Adapted from seminar
    example.
    Input:
        frames - list, list of filenames of the sequence of frames
        path - str, path to directory with frames
        lk_params - dict, parameters for opencv Lucas-Kanade method. Used for
        calculation of true velocities
        custom_lk_params - dict, params for pyr_lucas_kanade method
        visualize - bool, shows frames with special points movement if true
    Output:
        epes - list, list of EPE for each frame
        mean_epe - float, average EPE for all frames
    '''
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7,
                          blockSize=7)
    old_frame = cv2.imread(path + frames[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    color = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(old_frame)
    counter = 0
    epes = []
    while(1):
        if counter == len(frames):
            break
        frame = cv2.imread(path + frames[counter])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1 = pyr_lucas_kanade(old_gray, frame_gray, p0, **custom_lk_params)
        p1_cv, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0,
                                                  None, **lk_params)
        cur_epe = epe(p1-p0, p1_cv-p0, len(p1))
        epes.append(cur_epe)
        good_new = p1
        good_old = p0
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)),
                               5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        if visualize:
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new
        counter += 1
    print('Average (per frames) EPE:', np.mean(epes))
    if visualize:
        cv2.destroyAllWindows()
    return epes, np.mean(epes)


def calc_flow_on_video(name, lk_params, custom_lk_params):
    '''
    Function for EPE calculation on video and optical flow visualization on it.
    Adapted from seminar example.
    Input:
        name - str, path to video
        lk_params - dict, parameters for opencv Lucas-Kanade method. Used for
        calculation of true velocities
        custom_lk_params - dict, params for pyr_lucas_kanade method
    '''
    cap = cv2.VideoCapture(name)
    
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7,
                          blockSize=7)
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30,
                          (old_frame.shape[1], old_frame.shape[0]))
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    counter = 0
    epes = []
    while(1):
        ret, frame = cap.read()
        if not ret:
            # if we already got last frame
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1 = pyr_lucas_kanade(old_gray, frame_gray, p0, **custom_lk_params)
        p1_cv, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                                  p0, None, **lk_params)
        cur_epe = epe(p1-p0, p1_cv-p0, len(p1))
        epes.append(cur_epe)
        good_new = p1
        good_old = p0
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)),
                               5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        out.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new
        counter += 1
    out.release()
    print('Average (per frames) EPE:', np.mean(epes))
    cv2.destroyAllWindows()


def main():
    # Here is loop for calc overall EPE for SINTEL test dataset
    path_to_dirs = './test/final/'
    dirs = os.listdir(path_to_dirs)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    custom_lk_params = dict(win_size=51, pyr_size=3)
    epes = []
    for directory in dirs:
        frames = sorted(os.listdir(path_to_dirs + directory))
        path = path_to_dirs + directory + '/'
        buf_epes, ep = calc_epe(frames, path, lk_params,
                                custom_lk_params, visualize=False)
        epes += buf_epes
    print(np.mean(epes))
    custom_lk_params = dict(win_size=51, pyr_size=3)
    calc_flow_on_video('slow_traffic_small.mp4', lk_params, custom_lk_params)
    return 0


if __name__ == '__main__':
    main()
