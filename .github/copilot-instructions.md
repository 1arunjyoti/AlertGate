# Copilot instructions for AlertGate

## Commands

- Install dependencies: `pip install -r requirements.txt`
- Start the app: `python src/main.py`
- Run the camera check: `python test/camera_test.py`
- Run the Telegram bot check: `python test/telegram_bot_test.py`
- Run the YOLO detector smoke test: `python test/cat_detector.py`
- Build the container image: `docker compose build`
- Start the container stack: `docker compose up`
- Validate the compose file: `docker compose config --quiet`
- Audit dependencies: `pip-audit -r requirements.txt`

## Architecture

- `src/main.py` is the orchestrator. It loads `config/config.yaml`, applies env var overrides, wires the capture, motion, ROI, inference, temporal filtering, alerting, event storage, and web dashboard pieces, then runs the main frame loop.
- The runtime pipeline is: RTSP capture -> motion gate -> YOLO inference -> ROI filtering -> temporal voting -> alert delivery -> snapshot/event persistence -> dashboard update.
- `src/core/` holds the frame-processing primitives: RTSP capture, motion detection, ROI masking, temporal voting, and shared dataclasses.
- `src/alerts/` owns alert fan-out and snapshot creation. `AlertManager` sends notifications asynchronously and resets temporal state after an alert is emitted.
- `src/services/` contains persistence and external integrations. `ConfigStore` reads/writes YAML atomically, `EventStore` uses SQLite for event history, and `TelegramNotifier` sends Telegram messages/photos from environment variables.
- `src/web/api.py` exposes the FastAPI dashboard, MJPEG preview, WebSocket stats/events, and config mutation endpoints for ROI/classes.
- `src/web/templates/` and `src/web/static/` back the dashboard UI.

## Conventions

- Configuration is YAML-first. `config/config.yaml` is the source of truth, but runtime edits from the dashboard are persisted back through `ConfigStore`.
- Environment variables override config for secrets and deployment-specific values: `CAMERA_URL`, `DATABASE_PATH`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`.
- Detection thresholds are class-specific in `config/config.yaml`; code should preserve the per-class confidence and cooldown maps instead of flattening them.
- ROI coordinates are stored as normalized points and converted to pixel masks after the first frame arrives.
- The dashboard only encodes preview frames when clients are connected, so avoid moving expensive frame processing into the broadcast path.
- SQLite event writes use a dedupe index and WAL mode; preserve that behavior when changing event persistence.
- Many scripts in `test/` are standalone smoke checks rather than pytest tests; keep them runnable with `python path/to/script.py`.
