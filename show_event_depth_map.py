import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
import cupy as cp
from matplotlib.colors import LinearSegmentedColormap


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=0, dtype="uint8")
    p = [1 if i == 1 else 2 for i in p]
    img[y, x, :] = 0
    img[y, x, p] = 255
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def show_event_depth_map():
    try:
        dist_map = cv2.imread("EventDistanceMap.png")
        bar = cv2.imread("Data/Images/colorbar_grey.png")
        final = cv2.vconcat([dist_map, bar])
        final_name = "Event based Distance Map"
        cv2.namedWindow(final_name)
        cv2.moveWindow(final_name, 2750, 200)
        cv2.imshow(final_name, final)
        cv2.waitKey(0)
    except Exception as e:
        print("ERROR:", e)
        exit(-1)
    finally:
        cv2.destroyAllWindows()


def create_event_depth_map():
    try:
        event_files_left = sorted(glob.glob(os.path.join(args.input_dir[0], "*.npz")))
        event_files_right = sorted(glob.glob(os.path.join(args.input_dir[1], "*.npz")))
        events_l = np.load(event_files_left[133])
        events_r = np.load(event_files_right[133])
        shape = [int(x) for x in args.shape]
        img_l = render(shape=shape, **events_l)
        img_r = render(shape=shape, **events_r)
        resized_shape = (672, 376)
        img_left = cv2.resize(img_l, resized_shape, interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_r, resized_shape, interpolation=cv2.INTER_LINEAR)
        disp_mat = get_disparity_map(img_left, img_right, args.win_size)
        np.save("event_disp_mat.npy", disp_mat)
        create_depth_map(disp_mat, args.win_size)

    except Exception as e:
        print("ERROR:", e)
        exit(-1)
    finally:
        cv2.destroyAllWindows()


def get_match_2(x, y, left_img, right_img, w):
    side = 1 + 2 * w
    src_patch = np.zeros(shape=(side, side))
    tgt_patch = np.zeros(shape=(side, side))
    sads0 = []
    sads1 = []
    max_disp = 180
    # get src patch filled with data
    y_start = y - w if (y - w) >= 0 else 0
    y_end = y + w if (y + w) < left_img.shape[0] else left_img.shape[0] - 1
    x_start = x - w if (x - w) >= 0 else 0
    x_end = x + w if (x + w) < left_img.shape[1] else left_img.shape[1] - 1
    t = 0
    i = y_start
    while i <= y_end:
        j = x_start
        m = 0
        while j <= x_end:
            src_patch[t, m] = left_img[i, j]
            m += 1
            j += 1
        t += 1
        i += 1
    x_start = 0
    x_end = x_start + side
    while x_end < right_img.shape[1] - w:
        t = 0
        i = y_start
        if right_img[y, x_end - w].sum() > 0:
            while i <= y_end:
                j = x_start
                m = 0
                while j < x_end:
                    tgt_patch[t, m] = right_img[i, j]
                    m += 1
                    j += 1
                t += 1
                i += 1
            diff_patch = abs(src_patch - tgt_patch)
            sads0.append(diff_patch.sum())
        else:
            sads0.append(10e10)
        x_start += 1
        x_end += 1
    # backwards
    x_start = right_img.shape[1] - w
    x_end = x_start - side
    while x_end > w:
        t = 0
        i = y_start
        if right_img[y, x_end - w].sum() > 0:
            while i <= y_end:
                j = x_start
                m = side - 1
                while j > x_end:
                    tgt_patch[t, m] = right_img[i, j]
                    m -= 1
                    j -= 1
                t += 1
                i += 1
            diff_patch = abs(src_patch - tgt_patch)
            sads1.append(diff_patch.sum())
        else:
            sads1.append(10e10)
        x_start -= 1
        x_end -= 1
    opt0 = np.argmin(sads0)
    opt1 = len(sads1) - 1 - np.argmin(sads1)
    x_m0 = opt0 + w
    x_m1 = opt1 - w
    if x - x_m0 <= 0:
        x_m = x_m1
    elif x - x_m1 <= 0:
        x_m = x_m0
    else:
        x_m = x_m0 if sads0[opt0] < sads1[opt1] else x_m1
    return x_m


def get_disparity_map(left_img, right_img, w):
    init_val = 0.001
    max_disp = left_img.shape[1] // 6
    height, width = left_img.shape
    disp_mat = np.zeros_like(left_img) + init_val
    y = w
    while y < height - w:
        x = w
        while x < width - w:
            # find an event
            if left_img[y, x].sum() == 0:
                x += 1
                continue
            x_m = get_match_2(x, y, left_img, right_img, w)
            # calculate distance from disparity + uniqueness constraint
            d = x - x_m
            if d <= 0 or d > max_disp:
                d = 0.01
            if disp_mat[y, x] == init_val:
                disp_mat[y, x] = d
            # x += w // 2
            x += 1
        print("{}/{}".format(y, height))
        y += 1
    return disp_mat


def create_depth_map(disp_map, w):

    height = disp_map.shape[0]
    width = disp_map.shape[1]
    min_dist = 0.5
    max_dist = 2
    f = 700 * 0.0004  # ZED focal length in mm
    b = 120  # baseline in mm
    norm = matplotlib.colors.Normalize(vmin=min_dist, vmax=max_dist+2, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
    step = 1
    y = w
    dist_map = np.full_like(disp_map, fill_value=255)
    while y < height - w:
        x = w
        while x < width - w:

            d = disp_map[y, x]
            if d == 0.001:
                dist = max_dist+2
            else:
                dist = round(1.15 * b * f / d, 2)
            if dist > max_dist+2:
                dist = max_dist
            color = mapper.to_rgba(dist)[:-1]
            color = tuple([255 * x for x in color])
            cv2.rectangle(dist_map, (x - w, y - w), (x + w, y + w), color, -1)
            x += 1
        y += 1
    cv2.imshow("Distance Map", dist_map)
    cv2.imwrite("EventDistanceMap.png", dist_map)
    cv2.waitKey(0)


def show_distance_map(path_original, path_bar, path_map):
    final_name = 'Left image and its D-Map'
    cv2.namedWindow(final_name)
    cv2.moveWindow(final_name, 2750, 200)
    bar = cv2.imread(path_bar)
    zed_img = cv2.imread(path_original)
    zed_img_grey = cv2.imread(path_original, flags=cv2.IMREAD_GRAYSCALE)
    width = zed_img.shape[1]
    original = zed_img[:, :width // 2]
    d_map = cv2.imread(path_map)
    sbs = cv2.hconcat([original, d_map])
    filling = np.full(shape=[100, width - bar.shape[1], 3], fill_value=255, dtype=np.uint8)
    filling = cv2.hconcat([filling, bar])
    sbs = cv2.vconcat([sbs, filling])
    cv2.imshow(final_name, sbs)
    cv2.waitKey(0)
    # get edges from original image
    original_grey = zed_img_grey[:, :width // 2]
    original_blur = cv2.GaussianBlur(original_grey, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=original_blur, threshold1=100, threshold2=200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(d_map, contours, -1, (255, 255, 255), 1)
    cv2.destroyAllWindows()
    sbs = cv2.hconcat([original, d_map])
    sbs = cv2.vconcat([sbs, filling])
    cv2.imshow(final_name, sbs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="Data/Images/low_res2.png", help='Path to stereo side by side ZED image.')
    parser.add_argument("--input_dir", nargs=2, default=["Data/Events/Generated/Left", "Data/Events/Generated/Right"])
    parser.add_argument("--output_img", default="Data/Images/event_depth_img.png", help='file to save as depth matrix.')
    parser.add_argument("--shape", nargs=2, default=[160, 224])
    parser.add_argument("--win_size", default=12)
    args = parser.parse_args()
    # create_event_depth_map()
    show_event_depth_map()
    # disp_map = np.load("event_disp_mat.npy")
    # create_depth_map(disp_map, args.win_size)
