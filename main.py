import numpy as np
import cv2 as cv
from skin_finder import SkinFinder
import hand_finder as hf
from time import time
from model import Net
import torch
import threading
import webbrowser


class Predictor(threading.Thread):
    def run(self):
        global finder, frame, masks_merged, prediction
        global condition, hand_position
        global alternative
        global cut_merged
        with condition:
            while True:
                # t0 = time()
                condition.wait()
                masks_merged = finder.get_important_area(frame)

                if alternative:
                    x1, x2, s = hf.find_hand_alternative(masks_merged)
                else:
                    x1, x2, s = hf.find_hand(masks_merged)

                hand_position = ((x1, x2 + s), (x1 + s, x2))

                if s > 0:
                    cut = hf.cut_img(frame, x2, x1, s)
                    cut_merged = hf.cut_img(masks_merged, x2, x1, s)
                p = hf.predict(model, masks_merged, x2, x1, s)
                p = torch.max(p, 1)[1].item()

                if type(p) is int:
                    prediction = p
                # print(time()-t0)


def mouse_click(event, x, y, flags, userdata):
    global finder, frame
    if event == cv.EVENT_LBUTTONDOWN:
        finder.change_probing_point(x, y)
    if event == cv.EVENT_LBUTTONDBLCLK:
        finder.get_skin_color(frame)
    if event == cv.EVENT_MBUTTONDOWN:
        finder.add_background(frame)


def show_cats(arr):
    global catsshown
    if np.prod(arr) == 1 and not catsshown:
        webbrowser.open('https://imgflip.com/i/2jlwbi', new=2)
        catsshown = True


if __name__ == "__main__":
    loadingcat = [0, 0, 0, 0, 0, 0]
    catsshown = True

    cam = cv.VideoCapture(0)
    alternative = False  # when True uses predefined area

    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    device = torch.device('cpu')
    classes = ['C', 'L', 'fist', 'okay', 'palm', 'peace']
    print("loading model...")
    model = Net()
    model.load_state_dict(torch.load("ready_model_b.pt", map_location=device))
    model.eval()
    print("model loaded")

    # setting debug variables
    debug = False
    debug_height = 200  # height of debug windows
    debug_names = ["masks_merged", "skin_mask", "foreground_mask", "cut_merged"]

    # setting up display window
    display = cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("frame", 1000, 800)
    cv.moveWindow("frame", int(5/4*debug_height) + 100, 0)
    cv.setMouseCallback("frame", mouse_click)

    # setting up thread for predicting
    condition = threading.Condition()
    Predictor(daemon=True).start()
    prediction = 0
    hand_position = ((0, 0), (0, 0))
    finder = SkinFinder(width, height)
    masks_merged = np.zeros((width, height))
    cut_merged = masks_merged

    time_interval = 0.5  # minimal time between predictions
    t = 0  # time of last prediction
    while True:
        ret, frame = cam.read()
        key_input = cv.waitKey(1)
        if key_input == ord('b'):
            finder.add_background(frame)

        frame = finder.repair_brightness(frame)

        if key_input == ord('p'):
            finder.get_skin_color(frame)

        if key_input == ord('q'):
            break

        if key_input == ord('r'):
            finder.alpha = 1
            finder.kernel_size = 1

        if key_input == ord('d'):  # debug mode
            if debug:
                debug = False
                finder.hide_trackbars()
                for name in debug_names:
                    cv.destroyWindow(name)
            else:
                debug = True
                finder.show_trackbars()
                for i, name in enumerate(debug_names):
                    cv.namedWindow(name, cv.WINDOW_NORMAL)
                    cv.resizeWindow(name, int(5/4*debug_height), debug_height)
                    cv.moveWindow(name, 0, i*debug_height + 50)

        if key_input == ord('c'):
            finder.clear()

        if key_input == ord('h'):
            webbrowser.open('https://github.com/werkaaa/Python_project/blob/master/README.md', new = 2)

        if key_input == ord('m'):  # reset memes
            catsshown = False
            loadingcat = [0, 0, 0, 0, 0, 0]

        # changes between predefined area and hand searching
        if key_input == ord('a'):
            if alternative:
                alternative = False
            else:
                alternative = True

        if time()-t > time_interval:
            with condition:
                condition.notify()
            t = time()

        frame_to_show = np.copy(frame)
        frame_to_show = cv.rectangle(frame_to_show, hand_position[0], hand_position[1],
                                    (0, 255, 0), 5)
        frame_to_show = finder.place_marker(frame_to_show, hand_position[0], (0, 255, 0))
        frame_to_show = finder.place_marker(frame_to_show, hand_position[1], (0, 255, 0))
        frame_to_show = finder.place_marker(frame_to_show)
        cv.displayStatusBar('frame', "to get help press 'h'")
        text_position = (hand_position[0][0] + 20, hand_position[0][1]-20)
        cv.addText(frame_to_show, classes[prediction], text_position, nameFont="Times",
                   pointSize=30, color=(0, 255, 255))

        loadingcat[prediction] = 1
        show_cats(loadingcat)

        if debug:
            cv.imshow("masks_merged", masks_merged)
            cv.imshow("skin_mask", finder.skin_mask)
            cv.imshow("foreground_mask", finder.foreground_mask)
            if hand_position[0] != hand_position[1]:
                cv.imshow("cut_merged", cut_merged)
        cv.imshow("frame", frame_to_show)

    cam.release()
    cv.destroyAllWindows()
