import argparse
import sys
import matplotlib
import matplotlib.cm as cm
import cv2
import numpy as np
import cupy as cp
import numba

def plot_stereo_sad_disp(color=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="Data/Images/low_res2.png", help='Path to stereo side by side ZED image.')
    parser.add_argument("--shape", nargs=2, default=['672', '376'])
    parser.add_argument("--win_size", default=10)
    parser.add_argument("--color", default=1)
    args = parser.parse_args()
    shape = [int(x) for x in args.shape]
    shape = (shape[0], shape[1])
    img_path = args.img_path
    win_size = args.win_size
    color = args.color

    try:
        # convert a single stereo image to two right and left images
        zed_img = cv2.imread(img_path)
        width = zed_img.shape[1]
        img_left = zed_img[:, :width // 2]
        img_right = zed_img[:, width // 2:]
        left_name = "LEFT"
        right_name = "RIGHT"
        final_name = left_name+"   /   "+right_name
        cv2.namedWindow(final_name)
        cv2.moveWindow(final_name, 2750, 200)

        if shape is not (zed_img.shape[0], width):
            img_left = cv2.resize(img_left, shape, interpolation=cv2.INTER_LINEAR)
            img_right = cv2.resize(img_right, shape, interpolation=cv2.INTER_LINEAR)
        if color:
            cv2.setMouseCallback(final_name, get_disparity_3, param=[zed_img, win_size])
        else:
            zed_img = cv2.cvtColor(zed_img, cv2.COLOR_BGR2GRAY)
            # img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            cv2.setMouseCallback(final_name, get_disparity_2, param=[zed_img, win_size])
        while True:
            cv2.imshow(final_name, zed_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print("ERROR:", e)
        exit(-1)
    finally:
        cv2.destroyAllWindows()


def get_match_3(x, y, left_img, right_img, w):
    side = 1 + 2 * w
    src_patch = np.zeros(shape=(side, side, 3))
    tgt_patch = np.zeros(shape=(side, side, 3))
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
    x_m0 = opt0 + w
    x_m1 = opt1 - w
    if x - x_m0 <= 0:
        x_m = x_m1
    elif x - x_m1 <= 0:
        x_m = x_m0
    else:
        x_m = x_m0 if sads0[opt0] < sads1[opt1] else x_m1
    return x_m


def get_match_2(x, y, left_img, right_img, w):
    side = 1 + 2 * w
    src_patch = np.zeros(shape=(side, side))
    tgt_patch = np.zeros(shape=(side, side))
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
            src_patch[t, m] = left_img[i, j]
            m += 1
            j += 1
        t += 1
        i += 1
    x_start = x - w if (x - w) >= 0 else 0
    x_end = x_start + side
    while x_end < right_img.shape[1]:
        t = 0
        i = y_start
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
        x_start += 1
        x_end += 1
    # now going backward
    x_start = right_img.shape[1] - w
    x_end = x_start - side

    while x_end > w:
        t = side - 1
        i = y_start
        while i <= y_end:
            j = x_start
            m = side - 1
            while j < x_end:
                tgt_patch[t, m] = right_img[i, j]
                m -= 1
                j -= 1
            t -= 1
            i -= 1
        diff_patch = abs(src_patch - tgt_patch)
        sads1.append(diff_patch.sum())
        x_start -= 1
        x_end -= 1
        opt0 = np.argmin(sads0)
        opt1 = np.argmin(sads1)
    return opt0 if sads0[opt0] < sads1[opt1] else opt1


def get_disparity_3(event, x, y, flags, param):
    zed_img = param[0]
    w = param[1]
    width = zed_img.shape[1]
    left_img = zed_img[:, :width // 2]
    right_img = zed_img[:, width // 2:]
    red = (0, 0, 255)
    blue = (255, 0, 0)
    black = (0, 0, 0)
    green = (0, 255, 0)
    max_disp = left_img.shape[1] / 6
    if event == cv2.EVENT_LBUTTONDOWN:
        if x - w < 0 or x + w >= left_img.shape[1] or y - w < 0 or y + w >= left_img.shape[0]:
            return
        x_m = get_match_3(x, y, left_img, right_img, w)

        # calculate distance from disparity
        min_dist = 0.2
        max_dist = 2
        d = x - x_m
        if d <= 0 or d > max_disp:
            d = 0.01
            # x -= 5
            # x_m = get_match_3(x, y, left_img, right_img, w)
            # d = x - x_m

        f = 700 * 0.0004  # ZED focal length in mm
        b = 120  # baseline in mm
        dist = round(1.15 * b * f / d, 2)
        if dist > max_dist:
            dist = max_dist
        norm = matplotlib.colors.Normalize(vmin=min_dist, vmax=max_dist, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
        # if 0.33 <= dist <= 0.43:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), red, -1)
        # elif 0.73 <= dist <= 0.83:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), blue, -1)
        # elif 1.1 <= dist <= 1.3:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), green, -1)
        # else:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), black, -1)
        color = mapper.to_rgba(dist)[:-1]
        color = tuple([255*x for x in color])
        w = w//2
        cv2.rectangle(zed_img, (x - w, y - w), (x + w, y + w), color, -1)
        cv2.rectangle(zed_img, (width//2 + x_m - w, y - w), (width//2 + x_m + w, y + w), red, 2)
        cv2.putText(left_img, str(dist) + '[m]', (x - 2 * w, y + 3 * w), cv2.FONT_HERSHEY_SIMPLEX, 0.35, black, 1,
                    cv2.LINE_AA)


def get_disparity_2(event, x, y, flags, param):
    left_img = param[0]
    right_img = param[1]
    w = param[2]
    blue = (255, 0, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        if x - w < 0 or x + w >= left_img.shape[1] or y - w < 0 or y + w >= left_img.shape[0]:
            return
        x_m = get_match_2(x, y, left_img, right_img, w)
        cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), red, 2)
        cv2.rectangle(right_img, (x_m, y - w), (x_m + 2 * w, y + w), red, 2)
        # calculate distance from disparity
        d = x - x_m - w
        f = 1400 * 0.0004  # ZED focal length in mm
        b = 120  # baseline in mm
        dist = round(b * f / d, 2)
        cv2.putText(left_img, str(dist) + '[m]', (x - 2 * w, y + 3 * w), cv2.FONT_HERSHEY_SIMPLEX, 0.35, black, 1,
                    cv2.LINE_AA)


# TODO
def create_depth_mat():
    pass


if __name__ == "__main__":
    plot_stereo_sad_disp()
