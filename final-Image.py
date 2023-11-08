import cv2
from ultralytics import YOLO
import numpy as np
import os

folder = os.getcwd() + '/data/'
K = np.load(folder + 'K.npy')
dist = np.load(folder + 'dist.npy')

# object_size = (7, 8)

# objp = np.zeros((1, object_size[0] * object_size[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:object_size[0], 0:object_size[1]].T.reshape(-1, 2)

# Load the YOLOv8 model
model = YOLO('C:\\Users\\This\\Desktop\\ultralytics\\final-image\\weights\\best.pt')

 
clip = cv2.VideoCapture('left_output.avi')                                   # เปิดไฟล์ clip

while (True):
    ret, frame = clip.read()

    if ret:
        results = model(frame)                                               # ใช้ yolo ตรวจจับหนังสือในคลิป

        # Visualize the results on the frame
        annotated_frame = results[0].plot()                                   
        #cv2.imshow("YOLOv8 Inference", annotated_frame)                     # เปิดมาเป็นสีเทา

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x, y, w, h = map(int, box.xywh[0])

                center_x = x + (w - x) / 2
                center_y = y + (h - y) / 2

                top_right_x = x + w
                top_right_y = y


                bottom_left_x = x
                bottom_left_y = y + h

                top_left_x = x
                top_left_y = y

                obj_points = np.array([[0, 0, 0], [182, 0, 0], [182, 240, 0], [0, 240, 0]])                           # ขนาดวัตถุ

                obj_points = obj_points.astype('float32')

                image_points = np.array([[center_x, center_y], [center_x, center_y], [center_x, center_y], [center_x, center_y]])
                #image_points = image_points.astype('float32')
                #print("obj_points",obj_points)
                print("image_points",image_points)

                ret, rvecs, tvecs = cv2.solvePnP(obj_points, image_points, K, dist)
                tvecs = tvecs/1000
                x, y, z = tvecs.flatten()
                

                text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"
                cv2.putText(annotated_frame, text, (int(center_x), int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

clip.release()
cv2.destroyAllWindows()
