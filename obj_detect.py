import cv2
import mediapipe as mp
import time
import math
import threading
from pydub import AudioSegment
import numpy as np
import queue
import sounddevice as sd

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # Use 1 for better CPU performance
)

# Audio parameters
CHUNK_SIZE = 2048  # Larger chunk size for smoother playback
SAMPLE_RATE = 44100
CHANNELS = 1

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
    def __init__(self, audio_file, sample_rate=SAMPLE_RATE):
        self.audio = AudioSegment.from_mp3(audio_file)
        self.audio = self.audio.set_channels(CHANNELS)
        self.audio = self.audio.set_frame_rate(sample_rate)
        
        # Convert audio to numpy array
        self.audio_data = np.array(self.audio.get_array_of_samples()).astype(np.float32)
        self.audio_data /= np.max(np.abs(self.audio_data))
        
        self.position = 0
        self.volume = SmoothValue(-50.0, smoothing_factor=0.1)
        self.running = False
        self.audio_queue = queue.Queue(maxsize=3)
        
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        if self.running:
            if self.position + frames > len(self.audio_data):
                # Wrap around
                first_part = self.audio_data[self.position:]
                second_part = self.audio_data[:frames - len(first_part)]
                data = np.concatenate([first_part, second_part])
                self.position = len(second_part)
            else:
                data = self.audio_data[self.position:self.position + frames]
                self.position += frames
            
            # Apply smooth volume
            current_volume = self.volume.value
            gain = 10 ** (current_volume / 20)
            outdata[:] = (data * gain).reshape(-1, 1)
        else:
            outdata.fill(0)
    
    def start(self):
        self.running = True
        self.stream = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
        )
        self.stream.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def update_volume(self, distance_ratio):
        self.volume.update(-50 * distance_ratio)

def main():
    cap = cv2.VideoCapture(0)
    center_point = None
    calibration_done = False
    
    # Initialize audio
    try:
        audio_player = AudioPlayer("hurricane_sound.mp3")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal center_point
        if event == cv2.EVENT_LBUTTONDOWN:
            center_point = (x, y)
            print(f"Center point set at: {center_point}")
    
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback)
    
    print("Click anywhere in the window to set the reference point.")
    
    while center_point is None:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    
    # Start audio playback
    audio_player.start()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Use nose position as tracking point
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            h, w, _ = frame.shape
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            
            # Calculate distance
            distance = math.sqrt((nose_x - center_point[0])**2 + 
                               (nose_y - center_point[1])**2)
            max_distance = math.sqrt(w**2 + h**2) / 2.0
            distance_ratio = min(distance / max_distance, 1.0)
            
            # Update audio volume
            audio_player.update_volume(distance_ratio)
            
            # Draw visualization
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, center_point, 5, (255, 0, 0), -1)
            cv2.line(frame, (nose_x, nose_y), center_point, (0, 255, 255), 2)
        else:
            # If no person detected, gradually increase volume to max attenuation
            audio_player.update_volume(1.0)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    audio_player.stop()
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
