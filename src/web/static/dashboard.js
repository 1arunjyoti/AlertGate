class AlertGateDashboard {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.seenEventKeys = new Set();
    this.connectWebSocket();
    this.updateElements();
    this.loadRecentEvents();
  }
  // Establish WebSocket connection and set up event handlers
  connectWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProtocol}://${window.location.host}/ws`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.isConnected = true;
      this.updateConnectionStatus("connected", "Connected");
      console.log("WebSocket connected");
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // If it's an event, we want to add it immediately so we don't lose history
      if (data.type === "event") {
        this.addEvent(data.data);
      }
      // If it's stats, we only really care about drawing the very latest one
      else if (data.type === "stats") {
        if (!this.pendingStatsUpdate) {
          this.pendingStatsUpdate = true;
          // Using setTimeout avoids the background tab freezing issue of requestAnimationFrame
          // while still decoupling the WebSocket ingestion from the DOM render pipeline.
          setTimeout(() => {
            this.updateStats(data.data);
            this.pendingStatsUpdate = false;
          }, 50); // Small debounce window (20fps update cap for DOM)
        }
      }
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      this.updateConnectionStatus("error", "Disconnected");
      console.log("WebSocket disconnected");

      // Reconnect after 3 seconds
      setTimeout(() => {
        this.connectWebSocket();
      }, 3000);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.updateConnectionStatus("error", "Connection Error");
    };
  }
  // Load recent events from the server
  async loadRecentEvents() {
    try {
      const res = await fetch("/api/events");
      if (!res.ok) return;
      const data = await res.json();
      const events = data.events || [];
      // Add oldest first so latest ends up on top when inserted at beginning
      for (let i = events.length - 1; i >= 0; i--) {
        this.addEvent(events[i]);
      }
    } catch (e) {
      console.warn("Failed to load recent events", e);
    }
  }
  // Update connection status and text
  updateConnectionStatus(status, text) {
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    statusDot.className = `status-dot ${status}`;
    statusText.textContent = text;
  }
  // Update dashboard statistics
  updateStats(stats) {
    // Update FPS counter
    if (stats.fps) {
      document.getElementById("fpsCounter").textContent =
        `FPS: ${stats.fps.toFixed(1)}`;
    }

    // Update stats grid
    this.updateStatsGrid(stats);

    // Update temporal voting status
    if (stats.temporal_voting) {
      this.updateVotingStatus(stats.temporal_voting);
    }

    // Update motion status
    if (stats.motion) {
      this.updateMotionStatus(stats.motion);
    }
  }
  // Update the statistics grid with current stats
  updateStatsGrid(stats) {
    const statsGrid = document.getElementById("statsGrid");

    const statCards = [
      { label: "Total Detections", value: stats.total_detections || 0 },
      { label: "Alerts Sent", value: stats.alerts_sent || 0 },
      { label: "Current Frame", value: stats.frame_number || 0 },
      { label: "Uptime", value: this.formatUptime(stats.uptime || 0) },
    ];

    statsGrid.innerHTML = statCards
      .map(
        (card) => `
            <div class="stat-card">
                <div class="stat-value">${card.value}</div>
                <div class="stat-label">${card.label}</div>
            </div>
        `,
      )
      .join("");
  }
  // Update temporal voting status display
  updateVotingStatus(votingData) {
    const votingStatus = document.getElementById("votingStatus");

    const votingHtml = Object.entries(votingData)
      .map(([className, data]) => {
        const percentage = (data.current_votes / data.votes_required) * 100;
        const progressColor = percentage >= 100 ? "#28a745" : "#ffc107";

        return `
                <div class="voting-class">
                    <div class="voting-header">
                        <strong>${className.toUpperCase()}</strong>
                        <span>${data.current_votes}/${data.votes_required} votes</span>
                    </div>
                    <div class="voting-progress">
                        <div class="voting-bar" style="width: ${Math.min(percentage, 100)}%; background: ${progressColor}"></div>
                    </div>
                    <div class="voting-details">
                        Window: ${data.history_length}/${data.window_size} frames
                    </div>
                </div>
            `;
      })
      .join("");

    votingStatus.innerHTML = votingHtml;
  }
  // Update motion detection status display
  updateMotionStatus(motionData) {
    const motionDot = document.getElementById("motionDot");
    const motionText = document.getElementById("motionText");
    const motionArea = document.getElementById("motionArea");
    const motionContours = document.getElementById("motionContours");

    if (motionData.detected) {
      motionDot.className = "motion-dot active";
      motionText.textContent = "Motion Detected";
    } else {
      motionDot.className = "motion-dot";
      motionText.textContent = "No Motion";
    }

    motionArea.textContent = motionData.area || 0;
    motionContours.textContent = motionData.contours || 0;
  }
  // Add a new event to the events list
  addEvent(eventData) {
    const key = this.eventKey(eventData);
    if (this.seenEventKeys.has(key)) return;
    this.seenEventKeys.add(key);
    const eventsContainer = document.getElementById("eventsContainer");

    const eventHtml = `
            <div class="event-item">
                <div class="event-header">
                    <span>🚨 ${eventData.class_name} detected</span>
                    <span>${new Date(eventData.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="event-details">
                    <span class="event-badge ${eventData.confidence > 0.8 ? "badge-high" : "badge-med"}">Conf: ${(eventData.confidence * 100).toFixed(1)}%</span>
                    <span class="event-badge badge-neutral">Zone: ${eventData.zone || "Unknown"}</span>
                    <span class="event-badge badge-neutral">Frame: #${eventData.frame_number}</span>
                </div>
            </div>
        `;

    eventsContainer.insertAdjacentHTML("afterbegin", eventHtml);

    // Remove empty state if present
    const emptyState = document.getElementById("eventsEmptyState");
    if (emptyState && eventsContainer.contains(emptyState)) {
      emptyState.remove();
    }

    // Keep only last 20 events
    const events = eventsContainer.children;
    if (events.length > 20) {
      eventsContainer.removeChild(events[events.length - 1]);
    }
  }
  // Generate a unique key for an event to prevent duplicates
  eventKey(e) {
    if (e && typeof e.id === "number") return `id:${e.id}`;
    const ts = e && e.timestamp ? e.timestamp : "";
    const cls = e && e.class_name ? e.class_name : "";
    const fn = e && typeof e.frame_number !== "undefined" ? e.frame_number : "";
    const zone = e && e.zone ? e.zone : "";
    return `k:${ts}|${cls}|${fn}|${zone}`;
  }

  updateElements() {
    // Send heartbeat to keep WebSocket alive
    setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send("heartbeat");
      }
    }, 30000);
  }
  // Format uptime in hours and minutes
  formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
}

// Initialize dashboard when page loads
document.addEventListener("DOMContentLoaded", () => {
  new AlertGateDashboard();
  loadTargetClasses();
  loadRoiState();
});

// Config Logic
const DEFAULT_TARGET_CLASSES = [
  "person",
  "cat",
  "dog",
  "cow",
  "bicycle",
  "car",
  "motorcycle",
  "bus",
];

let availableTargetClasses = [...DEFAULT_TARGET_CLASSES];

function formatClassName(className) {
  return className
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderTargetClassCheckboxes(selectedClasses) {
  const container = document.getElementById("targetClassCheckboxes");
  if (!container) return;

  container.innerHTML = availableTargetClasses
    .map((className) => {
      const checked = selectedClasses.includes(className) ? "checked" : "";
      return `
        <label class="checkbox-label">
          <input type="checkbox" value="${className}" ${checked} />
          <span>${formatClassName(className)}</span>
        </label>
      `;
    })
    .join("");
}

async function loadTargetClasses() {
  try {
    const response = await fetch("/api/config/classes");
    if (response.ok) {
      const data = await response.json();
      const classes = data.classes || [];
      availableTargetClasses =
        data.available_classes && data.available_classes.length
          ? data.available_classes
          : DEFAULT_TARGET_CLASSES;
      renderTargetClassCheckboxes(classes);
    }
  } catch (e) {
    console.warn("Failed to load target classes", e);
    renderTargetClassCheckboxes(DEFAULT_TARGET_CLASSES);
  }
}

async function loadRoiState() {
  try {
    const response = await fetch("/api/config/roi/state");
    if (response.ok) {
      const data = await response.json();
      document.getElementById("toggleRoiEnabled").checked = data.enabled;
    }
  } catch (e) {
    console.warn("Failed to load ROI state", e);
  }
}

async function toggleRoiFeature(element) {
  const isEnabled = element.checked;
  try {
    const res = await fetch("/api/config/roi/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: isEnabled }),
    });
    if (!res.ok) {
      alert("Failed to update ROI state.");
      element.checked = !isEnabled; // revert UI
    }
  } catch (e) {
    console.error(e);
    alert("Error updating ROI state.");
    element.checked = !isEnabled;
  }
}

async function saveTargetClasses() {
  const selected = Array.from(
    document.querySelectorAll("#targetClassCheckboxes input:checked"),
  ).map((input) => input.value);

  try {
    const response = await fetch("/api/config/classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes: selected }),
    });

    if (response.ok) {
      showToast(
        "Target classes saved to config.yaml!<br>Please restart the bot for changes to take effect.",
        "success",
      );
    } else {
      const data = await response.json();
      showToast(
        "Error saving classes: " + (data.detail || "Unknown error"),
        "error",
      );
    }
  } catch (e) {
    showToast("Network error while saving classes: " + e.message, "error");
  }
}

// Toast Notification Logic
function showToast(message, type = "success") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = message;
  container.appendChild(toast);

  // Trigger layout before animating in
  void toast.offsetWidth;
  toast.classList.add("show");

  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300); // Wait for fade out
  }, 4000);
}

// ROI Editor Logic
let roiEditorActive = false;
let currentZoneType = null;
let polygonPoints = [];
let canvas = null;
let ctx = null;
let roiResizeHandler = null;

function toggleRoiEditor() {
  roiEditorActive = !roiEditorActive;
  canvas = document.getElementById("roiCanvas");
  const tools = document.getElementById("roiEditorTools");
  const btn = document.getElementById("btnEditRoi");

  if (roiEditorActive) {
    canvas.style.display = "block";
    tools.style.display = "flex";
    tools.style.flexWrap = "wrap";
    btn.innerHTML = "❌ Cancel Editing";
    initCanvas();
    if (!roiResizeHandler) {
      roiResizeHandler = () => {
        if (!roiEditorActive) return;
        resizeRoiCanvas();
        drawPolygon();
      };
      window.addEventListener("resize", roiResizeHandler);
    }
  } else {
    canvas.style.display = "none";
    tools.style.display = "none";
    btn.innerHTML = "✏️ Edit ROI Zones";
    currentZoneType = null;
    polygonPoints = [];
    if (roiResizeHandler) {
      window.removeEventListener("resize", roiResizeHandler);
      roiResizeHandler = null;
    }
  }
}

function resizeRoiCanvas() {
  const img = document.getElementById("videoFeed");
  if (!canvas || !img) return;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
}

function initCanvas() {
  resizeRoiCanvas();
  ctx = canvas.getContext("2d");

  canvas.onclick = (e) => {
    if (!currentZoneType) {
      showToast(
        "Please select whether to draw an Include or Exclude zone first.",
        "error",
      );
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const normX = x / canvas.width;
    const normY = y / canvas.height;

    polygonPoints.push([normX, normY]);
    drawPolygon();
  };
}

function startDrawing(type) {
  currentZoneType = type;
  polygonPoints = [];
  drawPolygon();
  showToast(
    `Started drawing ${type} zone. Click on the camera feed to place points.`,
    "success",
  );
}

function clearCurrentDrawing() {
  polygonPoints = [];
  drawPolygon();
}

function drawPolygon() {
  if (!ctx || !canvas) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (polygonPoints.length === 0) return;

  ctx.beginPath();
  ctx.moveTo(
    polygonPoints[0][0] * canvas.width,
    polygonPoints[0][1] * canvas.height,
  );

  for (let i = 1; i < polygonPoints.length; i++) {
    ctx.lineTo(
      polygonPoints[i][0] * canvas.width,
      polygonPoints[i][1] * canvas.height,
    );
  }

  if (polygonPoints.length > 2) ctx.closePath();

  ctx.lineWidth = 2;
  if (currentZoneType === "include") {
    ctx.strokeStyle = "#00ff00";
    ctx.fillStyle = "rgba(0, 255, 0, 0.3)";
  } else {
    ctx.strokeStyle = "#ff0000";
    ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
  }

  ctx.stroke();
  if (polygonPoints.length >= 3) {
    ctx.fill();
  }

  ctx.fillStyle = "#fff";
  for (let i = 0; i < polygonPoints.length; i++) {
    ctx.beginPath();
    ctx.arc(
      polygonPoints[i][0] * canvas.width,
      polygonPoints[i][1] * canvas.height,
      4,
      0,
      Math.PI * 2,
    );
    ctx.fill();
    ctx.stroke();
  }
}

async function saveRoi() {
  if (polygonPoints.length < 3) {
    showToast("Please define a polygon with at least 3 points.", "warning");
    return;
  }

  try {
    const response = await fetch("/api/config/roi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: currentZoneType, points: polygonPoints }),
    });

    if (response.ok) {
      showToast(
        "ROI saved successfully to config.yaml!<br>Please restart the bot for changes to take effect.",
        "success",
      );
      toggleRoiEditor();
    } else {
      const data = await response.json();
      showToast(
        "Error saving ROI: " + (data.detail || "Unknown error"),
        "error",
      );
    }
  } catch (e) {
    showToast("Network error while saving ROI: " + e.message, "error");
  }
}
