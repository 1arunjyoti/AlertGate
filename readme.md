# AlertGate 🛡️

**Real-time Animal & Human Detection System with Phone Camera Integration**

AlertGate transforms your spare smartphone into a powerful AI-driven security camera that detects animals and humans in real-time, sends instant notifications, and provides a modern web dashboard for monitoring.

## 🌟 Features

### Core Detection Capabilities
- **🎯 Multi-class Detection**: Cats, dogs, humans, and other animals using YOLO11n
- **📱 Phone Camera Integration**: Convert any smartphone into an IP camera via RTSP
- **🧠 Temporal Filtering**: K-of-N voting system to eliminate false alarms
- **🎭 Motion Gating**: CPU-efficient processing with background subtraction
- **🗺️ ROI Management**: Define include/exclude zones with polygon masking
- **⚡ Real-time Processing**: Optimized for low-end hardware (4-8GB RAM)

### Smart Alerting System
- **📲 Telegram Integration**: Instant notifications with snapshots
- **⏰ Configurable Cooldowns**: Per-class alert intervals to prevent spam
- **🎨 Rich Notifications**: Confidence scores, timestamps, and annotated images
- **📊 Event History**: Searchable log of all detection events

### Privacy & Security
- **🏠 Local Processing**: All AI inference runs on your hardware
- **🔒 Network Isolation**: No cloud dependencies for core functionality
- **📁 Local Storage**: Snapshots and clips stored on your device
- **🛡️ Secure Notifications**: End-to-end encrypted Telegram delivery

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
git clone https://github.com/yourusername/EdgeSentinel.git
cd AlertGate

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows: 

# Install dependencies
pip install -r requirements.txt
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
# Test camera connection
python test/camera_test.py

# Start EdgeSentinel
python src/main.py
```

**Expected Output:**
```
🚀 Starting AlertGate...
✅ Connected to RTSP stream: rtsp://Ip_Address:Port/h264_ucs.sdp
✅ YOLO11n model loaded successfully!
✅ Telegram bot connected: YourBotName
🌐 Web dashboard: http://localhost:port
🎯 Looking for: ['cat', 'dog', 'person']
```

## 🏗️ Architecture

```
AlertGate/
├── src/
│   ├── main.py                 # Application entry point
│   ├── core/                   # Core detection modules
│   │   ├── capture.py          # RTSP camera interface
│   │   ├── inference.py        # YOLO11n detection engine
│   │   ├── motion_gate.py      # Background subtraction
│   │   ├── roi.py              # Region of interest management
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