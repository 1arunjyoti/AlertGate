# AlertGate 

**Real-time Animal & Human Detection System with Phone Camera Integration**

AlertGate transforms your spare smartphone into a powerful AI-driven security camera that detects animals and humans in real-time, sends instant notifications, and provides a modern web dashboard for monitoring.

## 🌟 Features

### Core Detection Capabilities
- **Multi-class Detection**: Cats, dogs, humans, and other animals using YOLO11n
- **Phone Camera Integration**: Convert any smartphone into an IP camera via RTSP
- **Temporal Filtering**: K-of-N voting system to eliminate false alarms
- **Motion Gating**: CPU-efficient processing with background subtraction
- **ROI Management**: Define include/exclude zones with polygon masking
- **Real-time Processing**: Optimized for low-end hardware (4-8GB RAM)

### Smart Alerting System
- **Telegram Integration**: Instant notifications with snapshots
- **Configurable Cooldowns**: Per-class alert intervals to prevent spam
- **Rich Notifications**: Confidence scores, timestamps, and annotated images
- **Event History**: Searchable log of all detection events

### Privacy & Security
- **Local Processing**: All AI inference runs on your hardware
- **Network Isolation**: No cloud dependencies for core functionality
- **Local Storage**: Snapshots and clips stored on your device
- **Secure Notifications**: End-to-end encrypted Telegram delivery

## 🚀 Quick Start

### Prerequisites
- **Hardware**: 
  - Secondary smartphone (Android/iOS)
  - Laptop/PC with 4GB+ RAM
  - Stable Wi-Fi network (5GHz recommended)
- **Software**: 
  - Windows 10/11, MacOS, Linux
  - Python 3.8+ 
  - Git

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/1arunjyoti/AlertGate.git
cd AlertGate

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

If you prefer to run AlertGate without setting up Python locally, use Docker Compose.

```bash
# Clone the repository
git clone https://github.com/1arunjyoti/AlertGate.git
cd AlertGate

# Create your local environment file
cp .env.example .env
```

Edit `.env` with your RTSP camera URL and Telegram bot values:

```env
CAMERA_URL=rtsp://your-phone-ip:port/h264_ucs.sdp
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-telegram-chat-id
DATABASE_PATH=/app/media/alertgate.db
```

Then build and start the container:

```bash
docker compose up --build
```

The dashboard will be available at `http://localhost:8080`. Runtime data is stored in Docker volumes named `alertgate_media` and `alertgate_logs`, so it survives container rebuilds. The Docker image preloads the default `yolo11n.pt` model during build.

If you want to use a local model file instead of the one baked into the image, uncomment the model volume in `docker-compose.yml`:

```yaml
- ./yolo11n.pt:/app/yolo11n.pt:ro
```

### 2. Phone Setup (Camera)

**Android (Recommended):**
1. Install **IP Webcam** from Google Play Store
2. Configure settings:
   - Resolution: 1280x720
   - FPS: 15-20
   - Video encoder: H.264
   - Quality: 50%
3. Tap **"Start Server"**
4. Note the RTSP URL shown (e.g., `rtsp://Ip_Address:Port/h264_ucs.sdp`)

### 3. Telegram Bot Setup

1. **Create Bot:**
   - Message `@BotFather` on Telegram
   - Send `/newbot` and follow instructions
   - Save your bot token

2. **Get Chat ID:**
   - Message `@userinfobot` and send `/start`
   - Note your User ID (this becomes your chat_id)

### 4. First Run

```bash
# create your environment variable file
.env

# Test camera connection (optional)
python test/camera_test.py

# Test Telegram notifications (optional)
python test/telegram_test.py

# Start AlertGate
python src/main.py
```

**Expected Output:**
```
🚀 Starting AlertGate...
🌐 Web dashboard: http://localhost:8080
✅ Connected to RTSP stream: rtsp://Ip_Address:Port/h264_ucs.sdp
✅ YOLO11n model loaded successfully!
✅ Telegram bot connected: YourBotName
✅ Snapshot saved to ...
✅ Alert sent: cat (0.84)
```

## 🏗️ Architecture

```
AlertGate/
├── src/
│   ├── main.py                 # Application entry point
│   ├── core/                   # Core detection modules
│   │   ├── capture.py          # RTSP camera interface
│   │   ├── inference.py        # YOLO11n detection engine
│   │   ├── motion_gate.py      # Implements the motion detection
│   │   ├── roi.py              # Region of interest filtering
│   │   ├── temporal_filter.py  # K-of-N voting system
│   │   └── types.py            # Data structures
│   ├── services/               # External integrations
│   │   ├── telegram_service.py # Notification delivery
│   │   ├── storage_service.py  # File management
│   │   └── scheduler.py        # Time-based rules
│   └── web/                    # Dashboard interface
│       ├── api.py              # FastAPI endpoints
│       ├── templates/          # HTML templates
│       └── static/             # CSS/JS assets
├── config/
│   └── config.yaml             # Main configuration
├── media/                      # Generated content
│   ├── snapshots/              # Detection images
│   └── clips/                  # Video recordings
└── logs/                       # Application logs
```

### Data Flow

```
📱 Phone Camera (RTSP)
    ↓
🎥 Frame Capture
    ↓
🎭 Motion Detection Gate
    ↓ (if motion detected)
🧠 YOLO11n Inference
    ↓
🗺️ ROI Filtering
    ↓
🗳️ Temporal Voting (K-of-N)
    ↓ (if votes threshold met)
📸 Snapshot Creation
    ↓
📲 Telegram Notification
    ↓
🌐 Web Dashboard Update
```
