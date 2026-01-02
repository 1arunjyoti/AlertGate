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
        const wsUrl = `ws://${window.location.host}/ws`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus('connected', 'Connected');
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'stats') {
                this.updateStats(data.data);
            } else if (data.type === 'event') {
                this.addEvent(data.data);
            }
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus('error', 'Disconnected');
            console.log('WebSocket disconnected');
            
            // Reconnect after 3 seconds
            setTimeout(() => {
                this.connectWebSocket();
            }, 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('error', 'Connection Error');
        };
    }
    // Load recent events from the server
    async loadRecentEvents() {
        try {
            const res = await fetch('/api/events');
            if (!res.ok) return;
            const data = await res.json();
            const events = data.events || [];
            // Add oldest first so latest ends up on top when inserted at beginning
            for (let i = events.length - 1; i >= 0; i--) {
                this.addEvent(events[i]);
            }
        } catch (e) {
            console.warn('Failed to load recent events', e);
        }
    }
    // Update connection status and text
    updateConnectionStatus(status, text) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = text;
    }
    // Update dashboard statistics
    updateStats(stats) {
        // Update FPS counter
        if (stats.fps) {
            document.getElementById('fpsCounter').textContent = `FPS: ${stats.fps.toFixed(1)}`;
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
        const statsGrid = document.getElementById('statsGrid');
        
        const statCards = [
            { label: 'Total Detections', value: stats.total_detections || 0 },
            { label: 'Alerts Sent', value: stats.alerts_sent || 0 },
            { label: 'Current Frame', value: stats.frame_number || 0 },
            { label: 'Uptime', value: this.formatUptime(stats.uptime || 0) }
        ];
        
        statsGrid.innerHTML = statCards.map(card => `
            <div class="stat-card">
                <div class="stat-value">${card.value}</div>
                <div class="stat-label">${card.label}</div>
            </div>
        `).join('');
    }
    // Update temporal voting status display
    updateVotingStatus(votingData) {
        const votingStatus = document.getElementById('votingStatus');
        
        const votingHtml = Object.entries(votingData).map(([className, data]) => {
            const percentage = (data.current_votes / data.votes_required) * 100;
            const progressColor = percentage >= 100 ? '#28a745' : '#ffc107';
            
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
        }).join('');
        
        votingStatus.innerHTML = votingHtml;
    }
    // Update motion detection status display
    updateMotionStatus(motionData) {
        const motionDot = document.getElementById('motionDot');
        const motionText = document.getElementById('motionText');
        const motionArea = document.getElementById('motionArea');
        const motionContours = document.getElementById('motionContours');
        
        if (motionData.detected) {
            motionDot.className = 'motion-dot active';
            motionText.textContent = 'Motion Detected';
        } else {
            motionDot.className = 'motion-dot';
            motionText.textContent = 'No Motion';
        }
        
        motionArea.textContent = motionData.area || 0;
        motionContours.textContent = motionData.contours || 0;
    }
    // Add a new event to the events list
    addEvent(eventData) {
        const key = this.eventKey(eventData);
        if (this.seenEventKeys.has(key)) return;
        this.seenEventKeys.add(key);
        const eventsContainer = document.getElementById('eventsContainer');
        
        const eventHtml = `
            <div class="event-item">
                <div class="event-header">
                    <span>🚨 ${eventData.class_name} detected</span>
                    <span>${new Date(eventData.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="event-details">
                    Confidence: ${(eventData.confidence * 100).toFixed(1)}% | 
                    Zone: ${eventData.zone || 'Unknown'} | 
                    Frame: #${eventData.frame_number}
                </div>
            </div>
        `;
        
        eventsContainer.insertAdjacentHTML('afterbegin', eventHtml);
        
        // Keep only last 20 events
        const events = eventsContainer.children;
        if (events.length > 20) {
            eventsContainer.removeChild(events[events.length - 1]);
        }
    }
    // Generate a unique key for an event to prevent duplicates
    eventKey(e) {
        if (e && typeof e.id === 'number') return `id:${e.id}`;
        const ts = e && e.timestamp ? e.timestamp : '';
        const cls = e && e.class_name ? e.class_name : '';
        const fn = e && typeof e.frame_number !== 'undefined' ? e.frame_number : '';
        const zone = e && e.zone ? e.zone : '';
        return `k:${ts}|${cls}|${fn}|${zone}`;
    }

    updateElements() {
        // Send heartbeat to keep WebSocket alive
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('heartbeat');
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
document.addEventListener('DOMContentLoaded', () => {
    new AlertGateDashboard();
});
