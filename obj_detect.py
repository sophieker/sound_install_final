import cv2
import torch
import time
import math
import threading
from pydub import AudioSegment
import numpy as np
import queue
import sounddevice as sd

# --- Helper function for IoU ---
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# --- New Audio Playback Mechanism ---
class SmoothValue:
    def __init__(self, initial_value, smoothing_factor=0.1):
        self.value = initial_value
        self.target = initial_value
        self.smoothing_factor = smoothing_factor
    
    def update(self, target):
        self.target = target
        self.value += (self.target - self.value) * self.smoothing_factor
        return self.value

class AudioPlayer:
    def __init__(self, audio_file, sample_rate=44100):
        # Load and configure audio using pydub
        self.audio = AudioSegment.from_mp3(audio_file)
        self.audio = self.audio.set_channels(1)
        self.audio = self.audio.set_frame_rate(sample_rate)
        
        # Convert audio to numpy array (normalized)
        self.audio_data = np.array(self.audio.get_array_of_samples()).astype(np.float32)
        self.audio_data /= np.max(np.abs(self.audio_data))
        
        self.position = 0
        self.volume = SmoothValue(-50.0, smoothing_factor=0.1)
        self.running = False
        self.CHUNK_SIZE = 2048
        self.sample_rate = sample_rate

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        
        if self.running:
            # Wrap around if needed
            if self.position + frames > len(self.audio_data):
                first_part = self.audio_data[self.position:]
                second_part = self.audio_data[:frames - len(first_part)]
                data = np.concatenate([first_part, second_part])
                self.position = len(second_part)
            else:
                data = self.audio_data[self.position:self.position + frames]
                self.position += frames
            
            # Apply smooth volume gain
            current_volume = self.volume.value
            gain = 10 ** (current_volume / 20)
            outdata[:] = (data * gain).reshape(-1, 1)
        else:
            outdata.fill(0)

    def start(self):
        self.running = True
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=self.CHUNK_SIZE
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def update_volume(self, distance_ratio):
        # Update volume: 0 dB (no attenuation) when object is centered,
        # down to -50 dB when object is far (distance_ratio=1)
        self.volume.update(-50 * distance_ratio)

# --- Main script combining calibration & tracking with new audio ---
def main():
    # Load the YOLOv5 model
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True, trust_repo=True)
    model.conf = 0.5
    model.to('cpu')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Calibration phase to capture base objects
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
    
    # Set the center point via mouse click
    center_point = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal center_point
        if event == cv2.EVENT_LBUTTONDOWN:
            center_point = (x, y)
            print(f"Center point set at: {center_point}")
    
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
    
    # Capture a frame to determine the maximum possible distance
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame for audio configuration.")
        return
    height, width = frame.shape[:2]
    max_distance = math.sqrt(width**2 + height**2) / 2.0
    latest_distance = max_distance

    # Initialize the audio player using the new mechanism
    try:
        audio_player = AudioPlayer("hurricane_sound.mp3")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    audio_player.start()

    print("Entering tracking mode. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.xyxy[0]
        
        new_object_box = None
        # Identify any new object not part of the calibration boxes
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
            # Compute the center of the detected object and its distance from the set center
            obj_center = ((new_object_box[0] + new_object_box[2]) / 2,
                          (new_object_box[1] + new_object_box[3]) / 2)
            distance = math.sqrt((obj_center[0] - center_point[0])**2 + (obj_center[1] - center_point[1])**2)
            latest_distance = distance
            print(f"Distance from center: {distance:.2f} pixels")
            cv2.circle(frame, (int(obj_center[0]), int(obj_center[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(center_point[0]), int(center_point[1])), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(obj_center[0]), int(obj_center[1])), (int(center_point[0]), int(center_point[1])), (0, 255, 255), 2)
        else:
            # If no new object is detected, simulate the object moving away by increasing the distance gradually
            latest_distance = min(latest_distance + max_distance * 0.05, max_distance)
        
        # Update the audio volume based on the current distance ratio
        distance_ratio = min(latest_distance / max_distance, 1.0)
        audio_player.update_volume(distance_ratio)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    audio_player.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
