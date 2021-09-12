import matplotlib
import torch
import matplotlib.pyplot as plt, mpld3
import numpy as np
import glob
import cv2
import json
from vid2e.esim_torch.esim_torch import EventSimulator_torch


def create_event_file():
    esim_torch = EventSimulator_torch(contrast_threshold_neg=0.45,
                                      contrast_threshold_pos=0.45,
                                      refractory_period_ns=0)

    print("Loading images")
    for src in 'Left', 'Right':
        image_files = sorted(glob.glob("Data/Upsampled/"+src+"/imgs/*.png"))
        images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
        timestamps_s = np.genfromtxt("Data/Timestamps/timestamps.txt")
        timestamps_ns = (timestamps_s * 1e9).astype("int64")

        log_images = np.log(images.astype("float32") / 255 + 1e-4)

        # generate torch tensors
        print("Loading data to GPU")
        device = "cuda:0"
        log_images = torch.from_numpy(log_images).to(device)
        timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

        # generate events with GPU support
        print("Generating events")
        events = esim_torch.forward(log_images, timestamps_ns)
        events_np_dict = {k: v[:].cpu().numpy() for k, v in events.items()}
        events_dict = {'x': events_np_dict['x'].tolist(), 'y': events_np_dict['y'].tolist(),'t':events_np_dict['t'].tolist(),'p':events_np_dict['p'].tolist()}
        with open('Data/Events/events'+src+'.txt', 'w') as convert_file:
            convert_file.write(json.dumps(events_dict))
        print("Events written to "+src+" file successfully")


def show_events_video():
    with open("Data/Events/events.txt", "r") as f:
        events = json.load(f)
    print("Loading images")
    image_files = sorted(glob.glob("Data/Images/Left/*.png"))
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])
    imgs_num = len(images)
    events_num = len(events['x'])
    events_per_frame = events_num//imgs_num
    left_name = "Left Events"
    cv2.namedWindow(left_name)
    cv2.moveWindow(left_name, 3483, 200)

    while True:
        k = 0
        for i in range(imgs_num):
            # render events
            image = images[i]
            # print("Plotting")
            first_few_events = {k: v[events_per_frame*i:events_per_frame*(i+1)] for k,v in events.items()}
            image_color = np.stack([image,image,image],-1)
            image_color[first_few_events['y'], first_few_events['x'], :] = 0
            image_color[first_few_events['y'], first_few_events['x'], first_few_events['p']] = 255
            tmp = cv2.resize(image_color, (720, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(left_name, tmp)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
        if k & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def show_events_3d():
    with open("Data/Events/events.txt", "r") as f:
        events = json.load(f)

    begin = 30000
    end = 50000

    fig = plt.figure(figsize=(10, 7), dpi=80)
    ax = plt.axes(projection='3d')
    zdata = events['y'][begin:end]
    xdata = events['x'][begin:end]
    ydata = events['t'][begin:end]
    colors = events['p'][begin:end]
    colors = ["red" if color == 1 else "green" for color in colors]
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(200, -80)
    ax.scatter3D(xdata, ydata, zdata, marker='o', s=0.5, c=colors)

    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(3100,400,640, 545)

    plt.show()

if __name__ == "__main__":
    create_event_file()
    # show_events_3d()
