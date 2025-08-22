import cv2
import yaml
import time
import requests
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

class CatDetector:
    def __init__(self, config_path="config.yaml"):
        """Initialize the cat detector with configuration."""
        self.load_config(config_path)
        self.setup_logging()
        self.setup_model()
        self.setup_telegram()
        self.last_alert_time = 0
        self.frame_count = 0
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        load_dotenv()
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found!")
            raise

        # Override config with environment variables if they exist
        self.config['camera']['url'] = os.getenv('CAMERA_URL', self.config['camera']['url'])
        self.config['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', self.config['telegram']['bot_token'])
        self.config['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID', self.config['telegram']['chat_id'])
            
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config['logging']['level']
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """Load YOLO11n model."""
        try:
            model_name = self.config['detection']['model']
            print(f"🔄 Loading YOLO11n model: {model_name}")
            self.model = YOLO(model_name) 
            print(f"✅ Model loaded successfully!")
            
            self.class_names = self.model.names
            self.target_class_ids = []
            
            # Map target class names to IDs
            for target_class in self.config['detection']['target_classes']:
                for class_id, class_name in self.class_names.items():
                    if class_name.lower() == target_class.lower():
                        self.target_class_ids.append(class_id)
                        print(f"✅ Target class '{target_class}' mapped to ID {class_id}")
                        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def setup_telegram(self):
        """Setup Telegram bot configuration."""
        self.bot_token = self.config['telegram']['bot_token']
        self.chat_id = self.config['telegram']['chat_id']
        
        if self.bot_token == "YOUR_BOT_TOKEN_HERE":
            raise ValueError("❌ Please update bot_token in config.yaml!")
        if self.chat_id == "YOUR_CHAT_ID_HERE":
            raise ValueError("❌ Please update chat_id in config.yaml!")
            
        # Test bot connection
        self.test_telegram_connection()
        
    def test_telegram_connection(self):
        """Test Telegram bot connectivity."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info['ok']:
                    bot_name = bot_info['result']['first_name']
                    print(f"✅ Telegram bot connected: {bot_name}")
                else:
                    raise Exception("Bot API returned error")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Telegram connection failed: {e}")
            print("Please check your bot_token in config.yaml")
            raise
            
    def send_telegram_alert(self, detection_info, image_path=None):
        """Send alert to Telegram with optional image."""
        try:
            # Prepare message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = (
                f"CAT DETECTED! \n"
                f"Time: {timestamp}\n"
                f"Confidence: {detection_info['confidence']:.2f}\n"
                f"Frame: #{self.frame_count}"
            )

            # Send image if available and configured
            if image_path and self.config['alerts']['send_image']:
                success = self.send_telegram_photo(message, image_path)
            else:
                success = self.send_telegram_message(message)

            if success:
                print(f"✅ Telegram alert sent successfully")
            else:
                print(f"❌ Failed to send Telegram alert. Check logs for details.")

        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            
    def send_telegram_message(self, text):
        """Send text message to Telegram."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            self.logger.error(f"Telegram message failed: {response.status_code} {response.text}")
        return response.status_code == 200
        
    def send_telegram_photo(self, caption, image_path):
        """Send photo with caption to Telegram."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        with open(image_path, 'rb') as photo:
            data = {
                'chat_id': self.chat_id,
                'caption': caption
            }
            files = {'photo': photo}
            response = requests.post(url, data=data, files=files, timeout=30)
            if response.status_code != 200:
                self.logger.error(f"Telegram photo failed: {response.status_code} {response.text}")
            
        return response.status_code == 200
        
    def save_detection_image(self, frame, detections):
        """Save frame with detection boxes drawn."""
        try:
            # Create snapshots directory
            os.makedirs("snapshots", exist_ok=True)
            
            # Draw bounding boxes
            annotated_frame = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), 
                            (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (int(x1), int(y1-30)), 
                            (int(x1 + label_size[0]), int(y1)), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/cat_detection_{timestamp}_frame_{self.frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving detection image: {e}")
            return None
            
    def process_detections(self, results):
        """Process YOLO detection results."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf)
                    
                    # Check if it's a target class and meets confidence threshold
                    if (class_id in self.target_class_ids and 
                        confidence >= self.config['detection']['confidence']):
                        
                        bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                        class_name = self.class_names[class_id]
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        }
                        detections.append(detection)
                        
        return detections
        
    def should_send_alert(self):
        """Check if enough time has passed since last alert."""
        current_time = time.time()
        cooldown = self.config['alerts']['cooldown_seconds']
        return (current_time - self.last_alert_time) >= cooldown
        
    def run_detection(self):
        """Main detection loop."""
        # Setup camera capture
        camera_url = self.config['camera']['url']
        print(f"🔄 Connecting to camera: {camera_url}")
        
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
        
        if not cap.isOpened():
            raise Exception("❌ Failed to connect to camera stream")
            
        print(f"✅ Camera connected successfully!")
        print(f"🎯 Looking for: {self.config['detection']['target_classes']}")
        print(f"🔍 Confidence threshold: {self.config['detection']['confidence']}")
        print(f"⏰ Alert cooldown: {self.config['alerts']['cooldown_seconds']} seconds")
        print("\n🚀 Starting detection... (Press 'q' to quit)\n")
        
        process_every_n = self.config['detection']['process_every_n_frames']
        inference_size = self.config['detection']['inference_size']
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame, attempting to reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                    continue
                    
                self.frame_count += 1
                
                # Process every Nth frame for performance
                if self.frame_count % process_every_n != 0:
                    continue
                
                # Run inference
                results = self.model(frame, imgsz=inference_size, verbose=False)
                detections = self.process_detections(results)
                
                # Log detections to console
                if detections:
                    print(f"🎯 Frame #{self.frame_count}: Found {len(detections)} detection(s)")
                    for det in detections:
                        print(f"   - {det['class_name']}: {det['confidence']:.3f}")
                        
                    # Send alert if cooldown period has passed
                    if self.should_send_alert():
                        print(f"📱 Sending Telegram alert...")
                        
                        # Save detection image
                        image_path = self.save_detection_image(frame, detections)
                        
                        # Send alert
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        self.send_telegram_alert(best_detection, image_path)
                        
                        self.last_alert_time = time.time()
                    else:
                        remaining_cooldown = (self.config['alerts']['cooldown_seconds'] - 
                                           (time.time() - self.last_alert_time))
                        print(f"⏳ Alert cooldown active: {remaining_cooldown:.1f}s remaining")
                
                # Optional: Display frame (comment out for headless operation)
                display_frame = cv2.resize(frame, (800, 600))
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Cat Detection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Detection session ended")

def main():
    """Main function to run the detector."""
    try:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config" / "config.yaml"
        
        detector = CatDetector(config_path)
        detector.run_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
