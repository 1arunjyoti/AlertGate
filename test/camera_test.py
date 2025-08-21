import cv2
import time
import numpy as np
import os
from dotenv import load_dotenv

class FPSCounter:
    def __init__(self):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        
    def update(self):
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
        return self.fps

def main():
    
    load_dotenv()

    rtsp_url = os.getenv("CAMERA_URL")
    
    print(f"Connecting to: {rtsp_url}")
    
    # Configure OpenCV for RTSP
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optimize for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print("Error: Cannot connect to RTSP stream")
        print("Check:")
        print("1. Phone app is running and shows 'Server started'")
        print("2. IP address is correct")
        print("3. Both devices on same network")
        return
    
    fps_counter = FPSCounter()
    frame_count = 0
    
    print("Stream connected! Press 'q' to quit, 's' to save frame")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Warning: Failed to read frame")
            continue
            
        # Update FPS counter
        current_fps = fps_counter.update()
        frame_count += 1
        
        # Resize for display
        height, width = frame.shape[:2]
        if width > 800:
            scale_factor = 800 / width
            new_width = 800
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Draw FPS and frame info on image
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('RTSP Stream', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nStream ended. Total frames: {frame_count}")

if __name__ == "__main__":
    main()
