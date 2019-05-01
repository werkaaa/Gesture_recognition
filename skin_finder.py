import numpy as np
import cv2 as cv
from colorsys import rgb_to_hsv
import face_recognition



class SkinFinder:
    brig_norm_size = 10

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.background = None
        self.skin = None
        self.brightness_norm = None
        self.trackbars = None
        self.probe_idx = 0
        self.back_thresh = 30
        self.probing_points = [[300, 300]]
        self.thre_up = 3 * [30]
        self.thre_down = 3 * [30]
        self.back_thre_up = 3 * [30]
        self.back_thre_down = 3 * [30]
        self.kernel_size = 3
        self.alpha = 5

    def add_probing_point(self, coordinates: list):
        if coordinates[0] < self.width and coordinates[1] < self.height:
            self.probing_points.append(coordinates)
            print("added skin probing point")
        else:
            print("wrong coordinates")

    def show_trackbars(self):
        self.trackbars = [
            ["skin_Sat", 30, 70],
            ["skin_H_up", 20, 50],
            ["skin_V_up", 30, 70],
            ["skin_H_do", 20, 50],
            ["skin_V_do", 30, 70],
            ["back_Sat", 20, 50],
            ["back_H_up", 30, 70],
            ["back_V_up", 20, 50],
            ["back_H_do", 30, 70],
            ["back_V_do", 30, 120],
            ["kernel_size", 3, 10],
            ["alpha", 4, 10]
        ]

        cv.namedWindow("trackbars")
        for name, val, maximum in self.trackbars:
            cv.createTrackbar(name, "trackbars", val,
                              maximum, self.update_trackbars)

    def update_trackbars(self, a):
        if self.trackbars is None:
            return
        values = [cv.getTrackbarPos(name, "trackbars")
                  for name, a, b in self.trackbars]

        self.thre_up = [values[i] for i in [1, 0, 2]]
        self.thre_down = [values[i] for i in [3, 0, 4]]
        self.back_thre_up = [values[i] for i in [6, 5, 7]]
        self.back_thre_down = [values[i] for i in [8, 5, 9]]
        self.kernel_size = values[10]
        self.alpha = values[11]
        if self.kernel_size == 0:
            self.kernel_size = 1
        if self.alpha == 0:
            self.alpha = 1

    def get_skin_color(self, img):
        if self.skin is None:
            self.skin = []
        x, y = self.probing_points[self.probe_idx]
        color = img[y][x]
        r, g, b = color
        res = rgb_to_hsv(r, g, b)
        color = (res[0] * 180, res[1] * 255, res[2])
        if self.probe_idx < len(self.probing_points) - 1:
            self.probe_idx += 1
        print("added: ", color)
        self.skin.append(color)

    def find_skin(self, img):
        if self.skin is None:
            return np.ones((self.height, self.width), np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        part_res = len(self.skin) * [0]
        for i, color in enumerate(self.skin):
            low = [color[j] - self.thre_down[j] for j in range(3)]
            high = [color[j] + self.thre_up[j] for j in range(3)]
            part_res[i] = cv.inRange(img, np.array(low), np.array(high))
        result = np.zeros(img.shape[0:2], np.uint8)
        for i in range(len(self.skin)):
            result[part_res[i] != 0] = 255
        return result

    def rm_noise(self, img):
        kernel = (self.kernel_size, self.kernel_size)
        img = cv.erode(img, np.ones(kernel, np.uint8))
        kernel = (self.kernel_size * self.alpha, self.kernel_size * self.alpha)
        return cv.dilate(img, np.ones(kernel, np.uint8))

    def find_foreground(self, img):
        if self.background is None:
            return np.ones((self.height, self.width), dtype=np.uint8)

        low = self.background - self.back_thre_down
        low[np.less(low, 0)] = 0
        high = self.background + self.back_thre_up
        high[np.greater(high, 255)] = 255

        back = cv.inRange(img, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
        res = np.zeros((self.height, self.width), dtype=np.uint8)
        res[back == 0] = 255

        return res

    def add_background(self, img):
        print("Background added")
        self.background = img
        size = self.brig_norm_size
        brightness_norm = cv.cvtColor(self.background[0:size, 0:size, :], cv.COLOR_RGB2HSV)
        self.brightness_norm = np.mean(brightness_norm[:, :, 2])

    def repair_brightness(self, img):
        if self.brightness_norm is None:
            return img
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        size = self.brig_norm_size
        img_bri = np.mean(img[0:size, 0:size, 2])
        diff = img_bri - self.brightness_norm
        value = img[:, :, 2]
        value = value - diff
        value[value < 0] = 0
        value[value > 255] = 255
        value = np.array(value, np.uint8)
        img[:, :, 2] = value
        img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        return img

    def get_skin_mask(self, img):
        mask = self.find_skin(img)
        return self.rm_noise(mask)

    def get_foreground_mask(self, img):
        mask = self.find_foreground(img)
        return self.rm_noise(mask)

    def get_important_area(self, img):
        m1 = self.get_skin_mask(img)
        m2 = self.get_foreground_mask(img)

        res = np.zeros((self.height, self.width), np.uint8)
        idx = np.equal(m1, m2)
        idx[np.equal(m1, 0)] = False
        res[idx] = 255

        #faces = face_recognition.face_locations(img)
        #for left, bottom, right, top in faces:
        #    res[left:right, top:bottom] = 0

        return res


cam = cv.VideoCapture(0)

width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

finder = SkinFinder(width, height)
finder.show_trackbars()
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

    masks_merged = finder.get_important_area(frame)

    frame = cv.drawMarker(frame, tuple(finder.probing_points[finder.probe_idx]), (0, 0, 255))
    skin_mask = finder.get_skin_mask(frame)
    foreground = finder.get_foreground_mask(frame)
    cv.imshow("skin_mask", skin_mask)
    cv.imshow("foreground_no_noise", foreground)
    cv.imshow("masks_merged", masks_merged)
    cv.imshow("frame", frame)

cam.release()
cv.destroyAllWindows()
