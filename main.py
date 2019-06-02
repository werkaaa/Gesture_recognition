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
        global cut, cut_merged
        with condition:
            while True:
                # t0 = time()
                condition.wait()
                masks_merged = finder.get_important_area(frame)

                if alternative:
                    x1, x2, s = hf.find_hand_alternative(masks_merged)
                else:
                    x1, x2, s = hf.find_hand(masks_merged)
                hand_position = ((x1, x2), (x1+s, x2+s))

                if s>0:
                    cut = hf.cut_img(frame, x2, x1, s)
                    cut_merged = hf.cut_img(masks_merged, x2, x1, s)
                p = hf.predict(model, masks_merged, x2, x1, s)
                p = torch.max(p, 1)[1].item()
                if type(p) is int:
                    prediction = p
                # print(time()-t0)

def show_cats(arr):
    global catsshown
    if(np.prod(arr)==1 and not catsshown):
        webbrowser.open('https://imgflip.com/i/2jlwbi', new=2)
        catsshown = True


if __name__ == "__main__":
    time_interval = 0.5

    loadingcat = [0,0,0,0,0,0]
    catsshown = True

    cam = cv.VideoCapture(0)
    alternative = False

    cut = cam

    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    device = torch.device('cpu')
    classes = ['C', 'L', 'fist', 'okay', 'palm', 'peace']
    print("loading model...")
    model = Net()
    model.load_state_dict(torch.load("ready_model_b.pt", map_location=device))
    model.eval()
    print("model loaded")

    display = cv.namedWindow("frame", cv.WINDOW_NORMAL)
    debug = False

    condition = threading.Condition()
    Predictor(daemon=True).start()
    prediction = 0
    hand_position = ((0, 0), (0, 0))
    finder = SkinFinder(width, height)

    masks_merged = np.zeros((width, height))
    cut_merged = masks_merged
    t = 0
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

        if key_input == ord('a'):
            x = int(input("x: "))
            y = int(input("y: "))
            finder.add_probing_point((x, y))

        if key_input == ord('d'):
            debug = True
            finder.show_trackbars()
            cv.namedWindow("masks_merged", cv.WINDOW_NORMAL)
            cv.namedWindow("skin_mask", cv.WINDOW_NORMAL)
            cv.namedWindow("foreground_mask", cv.WINDOW_NORMAL)
            cv.namedWindow("cut", cv.WINDOW_NORMAL)
            cv.namedWindow("cut_merged", cv.WINDOW_NORMAL)

        if key_input == ord('c'):
            finder.clear()

        if key_input == ord('r'): #reset memes
            catsshown = False
            loadingcat = [0,0,0,0,0,0]

        if key_input == ord('l'): #change finding hand
            if(alternative):
                alternative = False
            else:
                alternative = True

        if time()-t > time_interval:
            with condition:
                condition.notify()
            t = time()

        frame = finder.place_marker(frame, hand_position[0], (0, 255, 0))
        frame = finder.place_marker(frame, hand_position[1], (0, 255, 0))
        frame = finder.place_marker(frame)
        cv.displayStatusBar("frame", classes[prediction])
        loadingcat[prediction] = 1
        show_cats(loadingcat)

        if debug:
            cv.imshow("masks_merged", masks_merged)
            cv.imshow("skin_mask", finder.skin_mask)
            cv.imshow("foreground_mask", finder.foreground_mask)
            cv.imshow("cut", cut)
            cv.imshow("cut_merged", cut_merged)
        cv.imshow("frame", frame)

