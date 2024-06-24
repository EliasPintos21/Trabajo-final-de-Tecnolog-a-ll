import numpy as np
import cv2
import argparse
import time
import sys
from utils import ARUCO_DICT, aruco_display

"""
python realAumCAM.py --type DICT_5X5_100 --camera true
"""

def overlay_image(frame, overlay, position, size):
    overlay_resized = cv2.resize(overlay, size)

    x, y = position
    h, w = overlay_resized.shape[:2]

    # Ensure the overlay is within the frame bounds
    if x + w > frame.shape[1]:
        w = frame.shape[1] - x
        overlay_resized = overlay_resized[:, :w]

    if y + h > frame.shape[0]:
        h = frame.shape[0] - y
        overlay_resized = overlay_resized[:h]

    if overlay_resized.shape[2] == 4:  # If overlay has alpha channel
        alpha_mask = overlay_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_mask * overlay_resized[:, :, c] +
                                      (1.0 - alpha_mask) * frame[y:y+h, x:x+w, c])
    else:  # If overlay does not have alpha channel
        frame[y:y+h, x:x+w] = overlay_resized

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
    video = cv2.VideoCapture(0)
    time.sleep(2.0)
else:
    if args["video"] is None:
        print("[Error] Video file location is not provided")
        sys.exit(1)
    video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# Cargar la imagen de superposición
overlay_img_path = "Images/dino.png"
overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = video.read()
    
    if ret is False:
        break

    h, w, _ = frame.shape
    width = 1000
    height = int(width * (h / w))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 24:  # Cambia esto al ID del marcador que quieras detectar
                corner = corners[i][0]
                top_left = corner[0]
                bottom_right = corner[2]
                size = (int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1]))
                position = (int(top_left[0]), int(top_left[1]))

                overlay_image(frame, overlay_img, position, size)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
