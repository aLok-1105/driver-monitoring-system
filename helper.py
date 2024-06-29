import cv2
import numpy as np
import time
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

current_label = None

#Left eyes indices
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] #16
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

#right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#Iris indices
RIGHT_IRIS=[ 469, 470, 471, 472 ]  
LEFT_IRIS=[ 474, 475, 476, 477  ]  

# lips indices
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

R_H_LEFT = [33]
R_H_RIGHT = [133]
L_H_LEFT = [362]
L_H_RIGHT = [263]


def distance(point_1, point_2):
    sum_of_squares = 0
    
    for coord1, coord2 in zip(point_1, point_2):
        squared_difference = (coord1 - coord2) ** 2
        sum_of_squares += squared_difference
    
    dist = sum_of_squares ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height) # convert ratios into pixel
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def denormalize_landmarks(landmarks, image_width, image_height):
    pixel_landmarks = []
    for lm in landmarks:
        pixel_x = int(lm.x * image_width)
        pixel_y = int(lm.y * image_height)
        pixel_landmarks.append((pixel_x, pixel_y))
    
    return pixel_landmarks


def iris_position(iris_center, right_point, left_point):
    center_right_dist = distance(iris_center, right_point)
    total_dist = distance(right_point, left_point)
    ratio = 0
    if total_dist[1]:
        ratio = center_right_dist/total_dist
    return ratio

def lips_position(lips_top, lips_bottom, right_point, left_point):
    top_bottom_dist = distance(lips_top, lips_bottom)
    right_left_dist = distance(right_point, left_point)
    ratio = top_bottom_dist/right_left_dist
    return ratio

def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # cv2.imshow('mask', mask)
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    r_max_x = max(right_eye_coords, key=lambda item: item[0])[0]
    r_min_x = min(right_eye_coords, key=lambda item: item[0])[0]
    r_max_y = max(right_eye_coords, key=lambda item: item[1])[1]
    r_min_y = min(right_eye_coords, key=lambda item: item[1])[1]

    l_max_x = max(left_eye_coords, key=lambda item: item[0])[0]
    l_min_x = min(left_eye_coords, key=lambda item: item[0])[0]
    l_max_y = max(left_eye_coords, key=lambda item: item[1])[1]
    l_min_y = min(left_eye_coords, key=lambda item: item[1])[1]

    cropped_right = eyes[r_min_y:r_max_y, r_min_x:r_max_x]
    cropped_left = eyes[l_min_y:l_max_y, l_min_x:l_max_x]

    return cropped_right, cropped_left

def positionEstimator(cropped_eye):
    if cropped_eye is None:
        return
    
    if cropped_eye.size == 0:
        return
    h, w = cropped_eye.shape
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)
    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]
    eye_position = pixelCounter(right_piece, center_piece, left_piece)
    return eye_position

def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    eye_parts = [right_part, center_part, left_part]

    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
    elif max_index == 1:
        pos_eye = 'CENTER'
    elif max_index == 2:
        pos_eye = 'LEFT'
    else:
        pos_eye = "Closed"
    return pos_eye 

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    avg_EAR = (left_ear + right_ear) / 2.0
    return avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def draw_iris(frame, landmarks, img_w, img_h):
    mesh_points = []
    for p in landmarks:
        point = np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
        mesh_points.append(point)
    mesh_points = np.array(mesh_points)
    (l_cx, l_cy), l_rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_rad = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
    c_left = np.array([l_cx, l_cy], dtype = np.int32)
    c_right = np.array([r_cx, r_cy], dtype = np.int32)
    cv2.circle(frame, c_left, int(l_rad), (100,255,0), 1, cv2.LINE_AA)
    cv2.circle(frame, c_right, int(r_rad), (100,255,0), 1, cv2.LINE_AA)

    eye_iris_ratio_left = iris_position(c_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT])
    eye_iris_ratio_right = iris_position(c_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT])
    avg_ratio = (eye_iris_ratio_left+eye_iris_ratio_right)/2
    return avg_ratio



def draw_lips(frame, landmarks, img_w, img_h):
    mesh_points = []
    for p in landmarks:
        point = np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
        mesh_points.append(point)
    mesh_points = np.array(mesh_points)

    # cv2.circle(frame, (mesh_points[0][0], mesh_points[0][1]), 2, (225, 0, 225), 1);
    for points in mesh_points[LIPS]:
        cv2.circle(frame, (points[0], points[1]), 2, (255, 0, 255), 1, -1)


def reset_counter(label):
    global current_label, start_time
    current_time = time.time()
    if current_label != label:
        current_label = label
        start_time = current_time
    elapsed_time = current_time - start_time
    return elapsed_time



