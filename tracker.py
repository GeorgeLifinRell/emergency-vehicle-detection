import math
import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO

class Tracker:
    def _init_(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



def video_analysis():
    distance1 = 0
    a_speed_kh1 = 0
    model = YOLO('models/yolo_v9.pt')
    cap = cv2.VideoCapture('video/highway.mp4')

    class_list = ['Ambulance', 'Label', 'Misc Vehicle', 'Siren', 'object']

    count = 0
    tracker = Tracker()
    down = {}
    up = {}
    counter_down = []
    counter_up = []

    red_line_y = 198
    blue_line_y = 268
    offset = 6

    # Create a folder to save frames
    if not os.path.exists('detected_frames'):
        os.makedirs('detected_frames')

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        # if count % 2 != 0:
        #     continue
        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        a = results[0].boxes.data
        a = a.detach().cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            # if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            #     down[id] = cy
            # if id in down:
            #     if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            #         cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            #         cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            #         counter_down.add(id)  # Add ID to set

            # if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            #     up[id] = cy
            # if id in up:
            #     if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            #         cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            #         cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            #         counter_up.add(id)  # Add ID to set
            if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                down[id]=time.time()   # current time when vehichle touch the first line
            if id in down:
            
                if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                    elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                if counter_down.count(id)==0:
                    counter_down.append(id)
                    distance = 10 # meters - distance between the 2 lines is 10 meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

                    
            #####going UP#####     
            if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                up[id]=time.time()
            if id in up:

                if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                    elapsed1_time=time.time() - up[id]
                # formula of speed= distance/time  (distance travelled and elapsed time) Elapsed time is It represents the duration between the starting point and the ending point of the movement.
                if counter_up.count(id)==0:
                    counter_up.append(id)      
                    distance1 = 10 # meters  (Distance between the 2 lines is 10 meters )
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        text_color = (0, 0, 0)  # Black color for text
        yellow_color = (0, 255, 255)  # Yellow color for background
        red_color = (0, 0, 255)  # Red color for lines
        blue_color = (255, 0, 0)  # Blue color for lines

        cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

        cv2.line(frame, (172, 198), (774, 198), red_color, 2)
        cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
        cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Save frame
        frame_filename = f'detected_frames/frame_{count}.jpg'
        # cv2.imwrite(frame_filename, frame)

        # out.write(frame)

        # cv2.imshow("frames", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #if cv2.waitKey(0) & 0xFF == 27:
            # break

    cap.release()
    cv2.destroyAllWindows()
    return distance1, a_speed_kh1
