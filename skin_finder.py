import numpy as np
import cv2 as cv
from colorsys import rgb_to_hsv

def draw_markers(img, points, color=(0, 0, 255), size=20):
    for point in points:
        img = cv.drawMarker(img, tuple(point), color, "MARKER_CROSS", size)
    return img


def get_color(img, p):
    color = img[p[0]][p[1]]
    r, g, b = color
    res = rgb_to_hsv(r, g, b)
    return res[0]*180, res[1]*255,res[2]


def find_skin(img, skin, thre_up, thre_down):
    if skin == []:
        return np.zeros(img.shape, np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    part_res = len(skin) * [0]
    for i,color in enumerate(skin):
        low = [color[j] - thre_down[j] for j in range(3)]
        high = [color[j] + thre_up[j] for j in range(3)]
        part_res[i] = cv.inRange(img, np.array(low), np.array(high))
    result = np.zeros(img.shape, np.uint8)
    for i in range(len(skin)):
        result[part_res[i] != 0] = 255
    return result

def rm_noise(img, kernel, alpha):
    img = cv.erode(img, np.ones((kernel, kernel),np.uint8))
    kernel *= alpha
    return cv.dilate(img, np.ones((kernel, kernel),np.uint8))


def rm_back(img, back):
    pass


cam = cv.VideoCapture(0)
mog = cv.createBackgroundSubtractorMOG2()
back_set = False
background = []
probing_points = [[100,100], [300, 300]]
probe_idx = 0
skin = []
thre_up = [5, 5, 5]
thre_down = [5, 5, 5]
trackbars = ["H_up", "S_up", "V_up", "H_do", "S_do", "V_do"]
cv.namedWindow("mask")
for bar in trackbars:
    cv.createTrackbar(bar, "mask", 50, 70, lambda x: 0)

while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    if not back_set:
        background = frame
        back_set = True

    if cv.waitKey(1) == ord('p'):
        color = get_color(frame, probing_points[probe_idx])
        if probe_idx < len(probing_points) - 1:
            probe_idx += 1
        print("added: ", color)
        skin.append(color)

    if cv.waitKey(1) == ord('c'):
        skin = []
        probe_idx = 0

    """
    if cv.waitKey(1) == ord('t'):
        print("skin colors:\n", skin, "\n")
        print("old threshold_down:\n", thre_down, "\n")
        print("old threshold_up:\n", thre_up)
        print("\nset threshold_down:")
        thre_down[0] = int(input("H:"))
        thre_down[1] = int(input("S:"))
        thre_down[2] = int(input("V:"))
        print("\nset threshold_up:")
        thre_up[0] = int(input("H:"))
        thre_up[1] = int(input("S:"))
        thre_up[2] = int(input("V:"))
    """

    if cv.waitKey(1) == ord('a'):
        x = int(input("x: "))
        y = int(input("y: "))
        probing_points=[probing_points,[x,y]]

    if cv.waitKey(1) == ord('q'):
        break

    mask = find_skin(frame, skin, thre_up, thre_down)
    cv.namedWindow("better mask")
    cv.createTrackbar("kernel", "better mask", 5, 10, lambda x: 0)
    cv.createTrackbar("alpha", "better mask", 5, 10, lambda x: 0)
    kernel = cv.getTrackbarPos("kernel", "better mask")
    alpha = cv.getTrackbarPos("alpha", "better mask")
    if kernel == 0:
        kernel = 1
    if alpha == 0:
        alpha = 1
    better_mask = rm_noise(mask, kernel, alpha)
    for i, bar in enumerate(trackbars[0:3]):
        thre_up[i] = cv.getTrackbarPos(bar,"mask")
    for i, bar in enumerate(trackbars[4:]):
        thre_down[i] = cv.getTrackbarPos(bar,"mask")

    if probe_idx < len(probing_points):
        frame = cv.drawMarker(frame, tuple(probing_points[probe_idx]), (0, 0, 255))
    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("better mask", better_mask)



cam.release()
cv.destroyAllWindows()
