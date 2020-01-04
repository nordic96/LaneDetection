import os

import cv2
import numpy as np

SRC_DIR = 'src'
SRC_FILENAME = os.path.join(SRC_DIR, 'test.mp4')
cap = cv2.VideoCapture(SRC_FILENAME)


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height - 50), (550, height - 50), (300, 170)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     3)
    return line_img


def canny(img):
    # Gray Scale conversion
    lane_image = np.copy(img)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    # Image Noise Filtering
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 50, 150)
    return canny_img


if __name__ == '__main__':
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        if ret:
            canny_image = canny(frame)
            masked_img = region_of_interest(canny_image)
            lines_in_img = cv2.HoughLinesP(masked_img,
                                           1.5,
                                           np.pi / 180,
                                           100,
                                           np.array([]),
                                           minLineLength=30,
                                           maxLineGap=5)
            line_img = display_lines(frame, lines_in_img)
            combo_img = cv2.addWeighted(frame, 0.9, line_img, 1, 1)

            cv2.imshow('testing', combo_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
