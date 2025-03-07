import cv2
import torch
import time
import math

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

center_point = None

def mouse_callback(event, x, y, flags, param):
    global center_point
    if event == cv2.EVENT_LBUTTONDOWN:
        center_point = (x, y)
        print(f"Center point set at: {center_point}")

model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.5
model.to('cpu')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Calibration phase: Ensure only static (base) objects are in view.")
print("Press 'c' to capture the current frame and calibrate base objects.")

calibration_boxes = []
calibration_done = False

while not calibration_done:
    ret, frame = cap.read()
    if not ret:
        break
    display_frame = frame.copy()
    cv2.putText(display_frame, "Calibration: Press 'c' to capture base objects", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Calibration", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.xyxy[0]
        calibration_boxes = []
        for det in detections:
            bbox = det[:4].tolist()
            calibration_boxes.append(bbox)
        print("Calibration complete. Detected base objects:")
        print(calibration_boxes)
        annotated_frame = results.render()[0]
        cv2.imshow("Calibration", annotated_frame)
        calibration_done = True
        time.sleep(1)

cv2.destroyWindow("Calibration")

print("Please click on the window to set the center point.")
cv2.namedWindow("Set Center")
cv2.setMouseCallback("Set Center", mouse_callback)

while center_point is None:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Set Center", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Set Center")

print("Entering tracking mode. Press 'q' to exit.")
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.xyxy[0]
    
    new_object_box = None
    for det in detections:
        bbox = det[:4].tolist()
        is_new = True
        for calib_box in calibration_boxes:
            if compute_iou(bbox, calib_box) > 0.3:
                is_new = False
                break
        if is_new:
            new_object_box = bbox
            break

    if new_object_box is not None and center_point is not None:
        obj_center = ((new_object_box[0] + new_object_box[2]) / 2,
                      (new_object_box[1] + new_object_box[3]) / 2)
        distance = math.sqrt((obj_center[0] - center_point[0])**2 + (obj_center[1] - center_point[1])**2)
        print(f"Distance from center: {distance:.2f} pixels")
        cv2.circle(frame, (int(obj_center[0]), int(obj_center[1])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(center_point[0]), int(center_point[1])), 5, (255, 0, 0), -1)
        cv2.line(frame, (int(obj_center[0]), int(obj_center[1])), (int(center_point[0]), int(center_point[1])), (0, 255, 255), 2)
    
    cv2.imshow("Tracking", frame)
    
    frame_count += 1
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
