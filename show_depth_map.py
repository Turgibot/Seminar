import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
import cupy as cp
from matplotlib.colors import LinearSegmentedColormap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="Data/Images/low_res2.png", help='Path to stereo side by side ZED image.')
    parser.add_argument("--bar_path", default="Data/Images/colorbar.png", help='Path to stereo side by side ZED image.')
    parser.add_argument("--output_mat", default="Data/Images/depth_img.png", help='file to save as depth matrix.')
    parser.add_argument("--shape", nargs=2, default=['160', '224'])
    parser.add_argument("--win_size", default=10)
    args = parser.parse_args()
    shape = [int(x) for x in args.shape]
    shape = (shape[0], shape[1])
    # show_dist_map('dist_mat.npy', args.win_size)
    # get_dist_mat(args.img_path, args.win_size, shape)
    # create_depth_mat(args.img_path, args.win_size)
    show_distance_map(args.img_path, args.bar_path, "DistanceMap0.png")


def show_dist_map(path_to_npy, w):
    dist_mat = np.load(path_to_npy)
    d_max = dist_mat.max()
    f = 700 * 0.0004  # ZED focal length in mm
    b = 120  # baseline in mm
    factor = 0.115
    dist_mat = factor * b * f / dist_mat
    black = np.array([0, 0, 0], dtype=int)
    min_dist = factor * b * f / d_max
    max_dist = 2

    norm = matplotlib.colors.Normalize(vmin=min_dist, vmax=max_dist, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
    step = w // 2
    d_map = np.zeros(list(dist_mat.shape) + [3], dtype=np.float32)
    y = w
    while y < d_map.shape[0] - w:
        x = w
        while x < d_map.shape[1] - w:
            dist = dist_mat[y, x]
            if dist > max_dist:
                dist = max_dist
            if dist < min_dist:
                color = black
            else:
                color = mapper.to_rgba(dist)[:-1]
                color = tuple([255 * x for x in color])
            cv2.rectangle(d_map, (x - step // 2, y - step // 2), (x + step // 2, y + step // 2), color, -1)
            x += step
        y += step

    cv2.imshow("Distance Map", d_map)
    cv2.waitKey(0)


def get_dist_mat(img_path, win_size, shape):
    try:
        # convert a single stereo image to two right and left images
        zed_img = cv2.imread(img_path)
        width = zed_img.shape[1]
        img_left = zed_img[:, :width // 2]
        img_right = zed_img[:, width // 2:]
        left_name = "LEFT"
        right_name = "RIGHT"
        # cv2.namedWindow(left_name)
        # cv2.namedWindow(right_name)
        # cv2.moveWindow(left_name, 2800, 200)
        # cv2.moveWindow(right_name, 3483, 200)

        # img_left = cv2.cvtColor(zed_img[:, :width // 2], cv2.COLOR_BGR2GRAY)
        # img_right = cv2.cvtColor(zed_img[:, width // 2:], cv2.COLOR_BGR2GRAY)
        shape = (shape[1], shape[0])
        # img_left = cv2.resize(img_left, shape, interpolation=cv2.INTER_LINEAR)
        # img_right = cv2.resize(img_right, shape, interpolation=cv2.INTER_LINEAR)
        dist_mat = get_disparity_map(img_left, img_right, win_size)
        np.save("dist_mat.npy", dist_mat)

    except Exception as e:
        print("ERROR:", e)


def get_match_3(x, y, left_img, right_img, w):
    side = 1 + 2 * w
    src_patch = np.zeros(shape=(side, side, 3))
    tgt_patch = np.zeros(shape=(side, side, 3))
    sads0 = []
    sads1 = []

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
            for k in range(3):
                src_patch[t, m, k] = left_img[i, j, k]
            m += 1
            j += 1
        t += 1
        i += 1
    x_start = 0
    x_end = x_start + side
    while x_end < right_img.shape[1]:
        t = 0
        i = y_start
        while i <= y_end:
            j = x_start
            m = 0
            while j < x_end:
                for k in range(3):
                    tgt_patch[t, m, k] = right_img[i, j, k]
                m += 1
                j += 1
            t += 1
            i += 1
        diff_patch = abs(src_patch - tgt_patch)
        sads0.append(diff_patch.sum())
        x_start += 1
        x_end += 1
    # backwards
    x_start = right_img.shape[1] - w
    x_end = x_start - side
    while x_end > w:
        t = 0
        i = y_start
        while i <= y_end:
            j = x_start
            m = side - 1
            while j > x_end:
                for k in range(3):
                    tgt_patch[t, m, k] = right_img[i, j, k]
                m -= 1
                j -= 1
            t += 1
            i += 1
        diff_patch = abs(src_patch - tgt_patch)
        sads1.append(diff_patch.sum())
        x_start -= 1
        x_end -= 1
    opt0 = np.argmin(sads0)
    opt1 = len(sads1) - 1 - np.argmin(sads1)
    x_m = (opt0 + w) if sads0[opt0] < sads1[opt1] else opt1 - w
    return x_m


def get_disparity_map(left_img, right_img, w):
    init_val = 0.001
    max_disp = left_img.shape[1] // 6
    height, width, _ = left_img.shape
    disp_mat = np.zeros((height, width), dtype=int) + init_val
    y = w
    while y < height-w:
        x = w
        while x < width - w:
            # if x-w < 0 or x+w >= left_img.shape[1] or y-w < 0 or y+w >= left_img.shape[0]:
            #     continue
            x_m = get_match_3(x, y, left_img, right_img, w)
            # calculate distance from disparity + uniqueness constraint
            d = x - x_m
            if d <= 0 or d > max_disp:
                d = 0.01
            if disp_mat[y, x] == init_val:
                disp_mat[y, x] = d
            x += w // 2
        print("{}/{}".format(y, height))
        y += w // 2
    return disp_mat


# TODO
def create_depth_mat(img_path, w):
    zed_img = cv2.imread(img_path)
    width = zed_img.shape[1]
    img_left = zed_img[:, :width // 2]
    img_right = zed_img[:, width // 2:]
    # left_name = "LEFT"
    # right_name = "RIGHT"
    # final_name = left_name+"   /   "+right_name
    # cv2.namedWindow(final_name)
    # cv2.moveWindow(final_name, 2750, 200)
    height = img_left.shape[0]
    width = img_left.shape[1]
    min_dist = 0.2
    max_dist = 2
    f = 700 * 0.0004  # ZED focal length in mm
    b = 120  # baseline in mm
    norm = matplotlib.colors.Normalize(vmin=min_dist, vmax=max_dist, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
    step = w // 2
    y = w
    dist_map = np.zeros_like(img_left)
    while y < height - w:
        x = w
        while x < width - w:
            x_m = get_match_3(x, y, img_left, img_right, w)
            d = x - x_m
            if d <= 0:
                dist = max_dist
            else:
                dist = round(1.15 * b * f / d, 2)
            if dist > max_dist:
                dist = max_dist
            color = mapper.to_rgba(dist)[:-1]
            color = tuple([255 * x for x in color])
            cv2.rectangle(dist_map, (x - w, y - w), (x + w, y + w), color, -1)
            # cv2.imshow("Distance Map", dist_map)
            x += w // 2
        y += w // 2
    cv2.imshow("Distance Map", dist_map)
    cv2.imwrite("EventDistanceMap0.png", dist_map)
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

def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    colors = [[color[2], color[1], color[0]] for color in colors]
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig = plt.figure(figsize=(8, 2))
    x_ticks = [x for x in range(1, 11)]
    x_labels = [round(x*0.2, 1) for x in range(1, 11)]
    plt.yticks([],[])
    plt.xticks(x_ticks, x_labels)
    # ax[1].set_xticks(x_ticks)
    # ax[1].set_xticklabels(x_labels)
    plt.imshow([colors], extent=[0, 10, 0, 1])
    # ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.savefig("colorbar.png")

if __name__ == "__main__":
    main()

