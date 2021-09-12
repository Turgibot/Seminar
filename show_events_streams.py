import argparse
import glob

import cv2
import numpy as np
import os


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=0, dtype="uint8")
    p = [1 if i == 1 else 2 for i in p]
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--input_dir", nargs=2, default=["Data/Events/Generated/Left", "Data/Events/Generated/Right"])
    parser.add_argument("--shape", nargs=2, default=[160, 224])
    args = parser.parse_args()

    event_files_left = sorted(glob.glob(os.path.join(args.input_dir[0], "*.npz")))
    event_files_right = sorted(glob.glob(os.path.join(args.input_dir[1], "*.npz")))
    events_l = np.load(event_files_left[0])
    events_r = np.load(event_files_right[0])
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
    cv2.imshow(final_name, final)
    cv2.waitKey(20)

    for r, l in zip(event_files_left[1:], event_files_right[1:]):
        events_l = np.load(l)
        events_r = np.load(r)
        img_l = render(shape=shape, **events_l)
        img_r = render(shape=shape, **events_r)
        img_left = cv2.resize(img_l, resized_shape, interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_r, resized_shape, interpolation=cv2.INTER_LINEAR)
        final = cv2.hconcat([img_left, img_right])
        cv2.imshow(final_name, final)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
