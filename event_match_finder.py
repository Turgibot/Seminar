import argparse
import glob
import os
import sys
import matplotlib
import matplotlib.cm as cm
import cv2
import numpy as np
import cupy as cp
import numba


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=0, dtype="uint8")
    p = [1 if i == 1 else 2 for i in p]
    img[y, x, :] = 0
    img[y, x, p] = 255
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



def plot_stereo_sad_disp(color=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="Data/Images/low_res2.png", help='Path to stereo side by side ZED image.')
    parser.add_argument("--input_dir", nargs=2, default=["Data/Events/Generated/Left", "Data/Events/Generated/Right"])
    parser.add_argument("--shape", nargs=2, default=[160, 224])
    parser.add_argument("--win_size", default=12)
    parser.add_argument("--color", default=1)
    args = parser.parse_args()
    shape = [int(x) for x in args.shape]
    shape = (shape[0], shape[1])
    img_path = args.img_path
    win_size = args.win_size
    color = args.color

    try:
        event_files_left = sorted(glob.glob(os.path.join(args.input_dir[0], "*.npz")))
        event_files_right = sorted(glob.glob(os.path.join(args.input_dir[1], "*.npz")))
        events_l = np.load(event_files_left[133])
        events_r = np.load(event_files_right[133])
        shape = [int(x) for x in args.shape]
        img_l = render(shape=shape, **events_l)
        img_r = render(shape=shape, **events_r)
        left_name = "LEFT"
        right_name = "RIGHT"
        final_name = left_name + "   /   " + right_name
        resized_shape = (672, 376)
        cv2.namedWindow(final_name)
        cv2.moveWindow(final_name, 2750, 200)
        img_left = cv2.resize(img_l, resized_shape, interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_r, resized_shape, interpolation=cv2.INTER_LINEAR)
        final = cv2.hconcat([img_left, img_right])
        cv2.setMouseCallback(final_name, get_disparity_2, param=[final, win_size])

        while True:
            cv2.imshow(final_name, final)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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
    while x_end < right_img.shape[1]-w:
        t = 0
        i = y_start
        if right_img[y, x_end-w].sum() > 0:
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
        if right_img[y, x_end-w].sum() > 0:
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


def get_disparity_2(event, x, y, flags, param):
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
        if x>=width//2:
            return
        if left_img[y, x].sum() == 0:
            return
        if x - w < 0 or x + w >= left_img.shape[1] or y - w < 0 or y + w >= left_img.shape[0]:
            return
        x_m = get_match_2(x, y, left_img, right_img, w)

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
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        # if 0.33 <= dist <= 0.43:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), red, -1)
        # elif 0.73 <= dist <= 0.83:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), blue, -1)
        # elif 1.1 <= dist <= 1.3:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), green, -1)
        # else:
        #     cv2.rectangle(left_img, (x - w, y - w), (x + w, y + w), black, -1)
        color = mapper.to_rgba(dist)[:-1]
        color = tuple([255 * x for x in color])
        w = w // 2
        cv2.rectangle(zed_img, (x - w, y - w), (x + w, y + w), color, -1)
        cv2.rectangle(zed_img, (width // 2 + x_m - w, y - w), (width // 2 + x_m + w, y + w), blue, 2)
        cv2.putText(left_img, str(dist) + '[m]', (x - 2 * w, y + 3 * w), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1,
                    cv2.LINE_AA)



# TODO
def create_depth_mat():
    pass


if __name__ == "__main__":
    plot_stereo_sad_disp()
