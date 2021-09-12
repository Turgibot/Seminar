import math
import sys
import time

import cv2


# TODO get configuration from json file instead of argv
def main():
    if len(sys.argv) < 2:
        print("Image path is required")
        exit(-1)
    img_path = sys.argv[1]
    try:
        a = int(sys.argv[2])
        b = int(sys.argv[3])
    except:
        a = 20
        b = 10
        print("Elliptical movement axis are set to default. a = {}, b = {}".format(a, b))
    try:
        speed = float(sys.argv[4])
    except:
        speed = 0.05
        print("Rotation peed parameter is set to default. speed= {} radians per frame".format(speed))
    try:
        duration = int(sys.argv[5])
    except:
        duration = 2
        print("Recording duration is set to default. duration = {} seconds".format(duration))
    try:
        fps = int(sys.argv[6])
    except:
        fps = 60
        print("video FPS is set to default. FPS = {}".format(fps))
    try:
        width_resize = int(sys.argv[7])
        height_resize = int(sys.argv[8])
    except:
        width_resize = 240
        height_resize = 180
        print("video frame size is set to default. size =({},{})".format(width_resize, height_resize))
    resized_shape = (width_resize, height_resize)
    try:
        show = int(sys.argv[9])
    except:
        show = 1
        print("Showing right image for debugging purposes")
    try:
        f = open(img_path, "r")
        # convert a single stereo image to two right and left images
        zed_img = cv2.imread(sys.argv[1])
        width = zed_img.shape[1]
        img_left = zed_img[:, :width // 2]
        img_right = zed_img[:, width // 2:]

        width = img_left.shape[1]
        height = img_left.shape[0]

        theta = 0
        imgs_l = []
        imgs_r = []
        stop = time.time() + duration
        index = 1
        print("Start stereo image saccades")
        while time.time() < stop:
            x = int(a * math.cos(theta))
            y = int(b * math.sin(theta))

            # crop original image according to new center
            img_left_crp = img_left[b + y:height - b + y, a - x:width - a - x]
            img_right_crp = img_right[b + y:height - b + y, a - x:width - a - x]
            imgs_l.append(img_left_crp)
            imgs_r.append(img_right_crp)
            # show is set to True to display the right image
            if show:
                tmp = cv2.resize(img_left_crp, (320, 240), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Left Image", tmp)
                cv2.waitKey(1000 // fps)
            else:
                time.sleep(1 / fps)
            theta += speed
        cv2.destroyAllWindows()
        # convert image list to videos
        print("Converting images to video")
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        left_video = cv2.VideoWriter('Data/Videos/left' + str(time.time()) + '.avi', fourcc, fps, resized_shape)
        right_video = cv2.VideoWriter('Data/Videos/right' + str(time.time()) + '.avi', fourcc, fps, resized_shape)
        for img in imgs_l:
            tmp = cv2.resize(img, resized_shape, interpolation=cv2.INTER_LINEAR)
            left_video.write(tmp)
        for img in imgs_r:
            tmp = cv2.resize(img, resized_shape, interpolation=cv2.INTER_LINEAR)
            right_video.write(tmp)

        print("Conversion is finished")

    except Exception as e:
        print("ERROR:", e)
        exit(-1)

    finally:
        f.close()
        cv2.destroyAllWindows()
        try:
            left_video.release()
            right_video.release()
        except:
            print("Error when releasing video writer")


if __name__ == "__main__":
    main()
