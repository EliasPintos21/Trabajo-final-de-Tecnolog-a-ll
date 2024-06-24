import cv2
import numpy as np

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image, superimpose_images=None):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            #cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            #cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            #cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            #cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            #cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            if superimpose_images and markerID in superimpose_images:
                overlay_image(image, superimpose_images[markerID], [topLeft, topRight, bottomRight, bottomLeft])
    
    return image

def overlay_image(background, overlay, pts):
    overlay_h, overlay_w = overlay.shape[:2]
    pts1 = np.array(pts, dtype='float32')
    pts2 = np.array([[0, 0], [overlay_w, 0], [overlay_w, overlay_h], [0, overlay_h]], dtype='float32')

    M = cv2.getPerspectiveTransform(pts2, pts1)
    overlay_warped = cv2.warpPerspective(overlay, M, (background.shape[1], background.shape[0]))

    mask = np.zeros_like(background, dtype='uint8')
    cv2.fillConvexPoly(mask, np.int32(pts1), (255, 255, 255))

    masked_background = cv2.bitwise_and(background, cv2.bitwise_not(mask))
    result = cv2.add(masked_background, overlay_warped)

    np.copyto(background, result)
