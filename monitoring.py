import cv2
import numpy as np
from math import floor
import time
from datetime import datetime
from helper import calculate_avg_ear, draw_iris, denormalize_landmarks, distance, reset_counter, LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS, LOWER_LIPS, UPPER_LIPS


st_time = time.time()-1
fix_time = 0
last_time = time.time()*1000
start_time = None
output = []

def monitoring(frame, facemesh_model, eye_idxs, state_tracker, thresholds):
    frame = cv2.flip(frame, 1)
    global fix_time, last_time, cnt, elapsed_time, region
    currTime = time.time() - st_time
    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh_model.process(rgbImg)
    img_h, img_w, _ = frame.shape
    face_2d = []
    face_3d = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    DROWSY_TIME_txt_pos = (10, int(img_h // 2 * 1.7))
    ALM_txt_pos = (10, int(img_h // 2 * 1.85))
    state_tracker["blink_rate"] = int((state_tracker["blinkCount"] / floor(currTime)) * 60)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        EAR, coordinates = calculate_avg_ear(landmarks, eye_idxs["left"], eye_idxs["right"], img_w, img_h) #coordinates are the 6 cordintates of eye in px of the image
        for eye_cord in coordinates:
            if eye_cord:
                for eye_point in eye_cord:
                    cv2.circle(frame, (eye_point), 1, (255, 255, 0), -1)
                    # cv2.polylines(frame, )
            continue
        
        avg_iris_eye_ratio = draw_iris(frame, landmarks, img_w, img_h)

        for idx, lm in enumerate(landmarks):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x*img_w, lm.y*img_h)
                    nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*3000)
                
                x, y = int(lm.x*img_w), int(lm.y*img_h)

                face_2d.append([x,y])
                face_3d.append([x,y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1*img_w

        cam_matrix = np.array([[focal_length, 0, img_h/2],
                                [0, focal_length, img_w/2],
                                [0,0,1]])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, tran_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR ,mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0]*360
        y = angles[1]*360
        z = angles[2]*360

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, tran_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*10))

        # process_conditions(x,y)

        cv2.line(frame, p1, p2, (250, 46, 57), 3)



        right_coords = denormalize_landmarks([landmarks[p] for p in RIGHT_EYE], img_w, img_h)
        left_coords = denormalize_landmarks([landmarks[p] for p in LEFT_EYE], img_w, img_h)

        right_coords_eye = np.array(right_coords, dtype=np.int32)
        left_coords_eye = np.array(left_coords, dtype=np.int32)
        
        right_coords_iris = denormalize_landmarks([landmarks[p] for p in RIGHT_IRIS], img_w, img_h)
        left_coords_iris = denormalize_landmarks([landmarks[p] for p in LEFT_IRIS], img_w, img_h)
        # lips_coords = denormalize_landmarks([landmarks[p] for p in LIPS], img_w, img_h)


        #YAWN Detection
        upper_lips_coords = denormalize_landmarks([landmarks[p] for p in UPPER_LIPS], img_w, img_h)
        lower_lips_coords = denormalize_landmarks([landmarks[p] for p in LOWER_LIPS], img_w, img_h)

        upper_lips_point = np.array(upper_lips_coords[9])
        lower_lips_point = np.array(lower_lips_coords[9])

        # Calculate the distance
        lips_dis_top_down = distance(upper_lips_point, lower_lips_point)

        if abs(lips_dis_top_down) > 10:
            cv2.putText(frame, 'Yawn', (img_w-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.001, (255, 255, 0), 2) 


        # Draw grid lines
        # cell_width = img_w // 5
        # cell_height = img_h // 2
        # for i in range(1, 5):
        #     cv2.line(frame, (i * cell_width, 0), (i * cell_width, img_w), (255, 0, 0), 1)
        # for i in range(1, 2):
        #     cv2.line(frame, (0, i * cell_height), (img_w, i * cell_height), (255, 0, 0), 1)
        

        right_coords_iris = np.array(right_coords_iris, dtype=np.int32)
        left_coords_iris = np.array(left_coords_iris, dtype=np.int32)
        
        frame_ratio = avg_iris_eye_ratio

        # print(EAR , state_tracker["blinkCount"])
        if EAR < thresholds["EAR_THRESH"]:
            end_time = time.perf_counter()
            state_tracker["DROWSY_TIME"] += end_time - state_tracker["start_time"]
            state_tracker["start_time"] = end_time
            state_tracker["COLOR"] = (0, 0, 255)
            # print(state_tracker["flag"])
            if state_tracker["flag"] and EAR < state_tracker["EAR_blink"]:
                state_tracker["blinkCount"] += 1
                state_tracker["flag"] = False   

            if state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                # state_tracker["play_alarm"] = True
                cv2.putText(frame, "WAKE UP! WAKE UP", ALM_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, state_tracker["COLOR"], 2)
        
        else:
            state_tracker["flag"] = True
            state_tracker["start_time"] = time.perf_counter()
            state_tracker["DROWSY_TIME"] = 0.0
            state_tracker["COLOR"] = (0, 255, 0)  # GREEN
            # state_tracker["play_alarm"] = False
        EAR_txt = f"EAR: {round(EAR, 2)}"
        DROWSY_TIME_txt = f"Blink Duration: {round(state_tracker['DROWSY_TIME'], 3)} Secs"
        cv2.putText(frame, EAR_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.02 , state_tracker["COLOR"], 2)
        cv2.putText(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.02 , state_tracker["COLOR"], 2)
        cv2.putText(frame, f"Blink Rate: {state_tracker['blink_rate']}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if x >= 9:
            if x <= 9:
                if y >= 10 and y <= 16:
                    region = '6'
                    elapsed_time = reset_counter('6'),
                    # cv2.putText(frame, '6', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                    #print(f"Counter 6: {elapsed_time:.2f} milliseconds")
            if y >= -5 and y < 5:
                region = '1'
                elapsed_time = reset_counter('1')
                # cv2.putText(frame, '1', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 1: {elapsed_time:.2f} milliseconds")
            elif y >= 5 and y < 11 and x >= 9:
                region = '2'
                elapsed_time = reset_counter('2')
                # cv2.putText(frame, '2', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 2: {elapsed_time:.2f} milliseconds")
            elif y >= 11 and y < 17 and x >= 9:
                region = '3'
                elapsed_time = reset_counter('3')
                # cv2.putText(frame, '3', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 3: {elapsed_time:.2f} milliseconds")
        else:
            if x >= 6 and y >= -5 and y < 5:
                region = '1'
                elapsed_time = reset_counter('1')
                # cv2.putText(frame, '1', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 1: {elapsed_time:.2f} milliseconds")
            if y >= -1 and y < 4:
                region = '4'
                elapsed_time = reset_counter('4')
                # cv2.putText(frame, '4', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 4: {elapsed_time:.2f} milliseconds")
            elif y >= 4 and y < 10:
                region = '5'
                elapsed_time = reset_counter('5')
                # cv2.putText(frame, '5', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 5: {elapsed_time:.2f} milliseconds")
            elif y >= 10 and y < 16:
                region = '6'
                elapsed_time = reset_counter('6')
                # cv2.putText(frame, '6', (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
                #print(f"Counter 6: {elapsed_time:.2f} milliseconds")
            
            # print(f"{elapsed_time:.2f}" )

        #Right Mirror
        if y >= -14 and y<-3: 
            region = 'Right Mirror'
            elapsed_time = reset_counter('7')
            # cv2.putText(frame, 'Right Mirror', (img_w//2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX,  2, (200, 200, 20), 3)
        
        #Left Mirror
        if y >= 16:
            region = 'Left Mirror'
            elapsed_time = reset_counter('8')
            # cv2.putText(frame, 'Left Mirror', (img_w//2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX,  2, (200, 200, 20), 3)
        
        #Right Shoulder
        if (y <-14) and (frame_ratio>=0.2).all():
            region = 'Right Shoulder'
            elapsed_time = reset_counter('9')
            # cv2.putText(frame, 'Right Shoulder', (img_w//2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX,  2, (200, 200, 20), 3)
        
        #Left Shoulder
        if (y >= 17) and (frame_ratio >= 0.88).all():
            region = 'Left Shoulder'
            elapsed_time = reset_counter('10')
            # cv2.putText(frame, 'Left Shoulder', (img_w//2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX,  2, (200, 200, 20), 3)
        
        #Dirstraction
        if x<2 :
            region = 'Distracted'
            elapsed_time = reset_counter('11')
            current_time = time.time()
            seconds = current_time - last_time
            fix_time += seconds
            last_time = current_time
        cv2.putText(frame, region,  (img_w // 2 - 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 20), 3)
        output.append((timestamp, EAR, state_tracker['blink_rate'], state_tracker['blinkCount'], state_tracker["DROWSY_TIME"], region, f"{elapsed_time:.6f}"))
    
    else:

        state_tracker["start_time"] = time.perf_counter()
        state_tracker["DROWSY_TIME"] = 0.0
        state_tracker["COLOR"] = (0, 255, 0)
        output.append((timestamp, 'NAN', state_tracker['blink_rate'], state_tracker['blinkCount'], state_tracker["DROWSY_TIME"], 'NAN', 'NAN'))
        
    
    return frame, output
