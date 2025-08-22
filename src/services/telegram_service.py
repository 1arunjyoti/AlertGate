import requests
from typing import Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self):
        """
        Reads configuration from environment variables:
          - TELEGRAM_BOT_TOKEN: str
          - TELEGRAM_CHAT_ID: str or int
        """
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not self.bot_token or not self.chat_id:
            raise ValueError("TelegramNotifier requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set in the environment.")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_text(self, text: str) -> bool:
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": str(self.chat_id),
            "text": text
        }
        try:
            r = requests.post(url, data=data, timeout=10)
            return r.ok
        except Exception:
            return False

    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        url = f"{self.base_url}/sendPhoto"
        data = {
            "chat_id": str(self.chat_id),
            "caption": caption or ""
        }
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                r = requests.post(url, data=data, files=files, timeout=30)
            return r.ok
        except Exception:
            return False

    def send_detection_alert(self, detection, snapshot_path: Optional[str] = None) -> bool:
        """
        detection: Detection dataclass (class_name, confidence, bbox, timestamp)
        snapshot_path: optional path to annotated image
        """
        ts = detection.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"Alert: {detection.class_name} {detection.confidence:.2f} at {ts}"

        if snapshot_path and os.path.exists(snapshot_path):
            return self.send_photo(snapshot_path, caption=msg)
        else:
            return self.send_text(msg)
