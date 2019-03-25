import numpy as np
import cv2 as cv
from colorsys import rgb_to_hsv

"""
def draw_markers(img, points, color=(0, 0, 255), size=20):
    for point in points:
        img = cv.drawMarker(img, tuple(point), color, "MARKER_CROSS", size)
    return img
"""


def get_color(img, p):
    color = img[p[1]][p[0]]
    r, g, b = color
    res = rgb_to_hsv(r, g, b)
    return res[0]*180, res[1]*255, res[2]


def find_skin(img, skin, thre_up, thre_down):
    if skin is None:
        return np.zeros(img.shape[0:2], np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    part_res = len(skin) * [0]
    for i, color in enumerate(skin):
        low = [color[j] - thre_down[j] for j in range(3)]
        high = [color[j] + thre_up[j] for j in range(3)]
        part_res[i] = cv.inRange(img, np.array(low), np.array(high))
        print(part_res[i][100,300], low, img[100,300,:],high)
    result = np.zeros(img.shape[0:2], np.uint8)
    for i in range(len(skin)):
        result[part_res[i] != 0] = 255
    print (result[100,300])
    return result


def rm_noise(img, kernel, alpha):
    img = cv.erode(img, np.ones((kernel, kernel),np.uint8))
    kernel *= alpha
    return cv.dilate(img, np.ones((kernel, kernel),np.uint8))


def find_foreground(img, back, threshold):
    out = np.empty(img.shape, np.uint8)
    cv.absdiff(img, back, out)
    res = np.zeros(img.shape[0:2], np.uint8)
    for i in range(3):
        res[out[:, :, i] > threshold] += 1
    res[res >= 2] = 255
    return res


def repair_brightness(img, norm, size):
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    norm = cv.cvtColor(norm, cv.COLOR_RGB2HSV)
    diff = img[0:size, 0:size, 2].astype(np.int8) - norm[0:size, 0:size, 2].astype(np.int8)
    diff = np.mean(diff)
    value = img[:, :, 2]
    value = value - diff
    value[value < 0] = 0
    value = np.array(value, np.uint8)
    img[:, :, 2] = value
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def update_trackbars(a):
    values = [cv.getTrackbarPos(bar, "trackbars") for bar in trackbar_names]
    global thre_up
    global thre_down
    global back_thresh
    global thre_down
    global kernel_size
    global alpha

    thre_up = values[0:3]
    thre_down = values[3:6]
    back_thresh = values[6]
    kernel_size = values[7]
    alpha = values[8]
    if kernel_size == 0:
        kernel_size = 1
    if alpha == 0:
        alpha = 1
    print(values)


def merge_masks(m1, m2, shape):
    res = np.zeros(shape, np.uint8)
    idx = np.equal(m1, m2)
    idx[np.equal(m1, 0)] = 0
    res[idx] = 255
    return res


cam = cv.VideoCapture(0)

width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
print(width, height)
background = None
skin = None
probe_idx = 0
back_thresh = 30
probing_points = [[100, 300]]
thre_up = [5, 5, 5]
thre_down = [5, 5, 5]
kernel_size = 3
alpha = 5
foreground_mask = np.ones((height, width), dtype=np.uint8) * 255

trackbar_names = ("H_up", "S_up", "V_up", "H_do", "S_do", "V_do", "back_thresh", "kernel_size", "alpha")
track_val = ((40, 70), (30, 70), (30, 70), (40, 70), (30, 70), (30, 70), (30, 70), (3, 10), (5, 10))
cv.namedWindow("trackbars")
for name, values in zip(trackbar_names, track_val):
    cv.createTrackbar(name, "trackbars", values[0], values[1], update_trackbars)

while True:
    ret, frame = cam.read()

    if cv.waitKey(1) == ord('b'):
        print("Background added")
        background = frame

    if background is not None:
        frame = repair_brightness(frame, background, 10)
        foreground_mask = find_foreground(frame, background, back_thresh)
        #cv.imshow("foreground_mask", foreground_mask)

    if cv.waitKey(1) == ord('p'):
        if skin == None:
            skin = []
        color = get_color(frame, probing_points[probe_idx])
        if probe_idx < len(probing_points) - 1:
            probe_idx += 1
        print("added: ", color)
        skin.append(color)

    if cv.waitKey(1) == ord('c'):
        skin = None
        probe_idx = 0

    if cv.waitKey(1) == ord('a'):
        x = int(input("x: "))
        y = int(input("y: "))
        probing_points=[probing_points,[x,y]]

    if cv.waitKey(1) == ord('q'):
        break

    skin_mask = find_skin(frame, skin, thre_up, thre_down)

    skin_no_noise = rm_noise(skin_mask, kernel_size, alpha)
    foreground_no_noise = rm_noise(foreground_mask, kernel_size, alpha)

    masks_merged = np.zeros((height, width))
    masks_merged = merge_masks(skin_no_noise, foreground_no_noise, (height, width))

    frame = cv.drawMarker(frame, tuple(probing_points[probe_idx]), (0, 0, 255))

    #cv.imshow("skin_mask", skin_mask)
    #cv.imshow("foreground_no_noise", foreground_no_noise)
    cv.imshow("masks_merged", masks_merged)
    cv.imshow("frame", frame)

    #cv.imshow("skin_no_noise", skin_no_noise)


cam.release()
cv.destroyAllWindows()
