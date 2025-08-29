// Toast Notification System - Enhanced with Real-time Dashboard Updates
class ToastManager {
    constructor() {
        this.container = this.createContainer();
        this.anomalies = [];
        this.interval = null;
        this.displayedAnomalies = { high: 0, medium: 0, low: 0 };
        this.totalDisplayed = 0;
    }

    createContainer() {
        const container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
        return container;
    }

    async loadAnomalies() {
        try {
            console.log('Fetching anomaly data from detailed_anomaly_report.json...');
            const response = await fetch('/anomaly-data/');
            this.anomalies = await response.json();
            console.log(`✅ Successfully loaded ${this.anomalies.length} vehicle anomalies from JSON file`);
            console.log('Sample anomaly data:', this.anomalies[0]);
            
            // Initialize dashboard statistics
            this.updateDashboardStats();
            return true;
        } catch (error) {
            console.error('❌ Error loading vehicle anomaly data:', error);
            this.showErrorToast('Failed to load vehicle anomaly data from detailed_anomaly_report.json');
            return false;
        }
    }

    updateDashboardStats() {
        // Calculate overall statistics for the chart and initial display
        const stats = { high: 0, medium: 0, low: 0, total: this.anomalies.length };
        
        this.anomalies.forEach(anomaly => {
            const severity = anomaly.severity.toLowerCase();
            if (severity === 'high') stats.high++;
            else if (severity === 'medium') stats.medium++;
            else if (severity === 'low') stats.low++;
        });

        // Update dashboard elements if they exist
        if (window.updateStatsDisplay) {
            window.updateStatsDisplay(stats);
        }
        if (window.updateChart) {
            window.updateChart(stats);
        }
    }

    startToastInterval() {
        if (this.interval) {
            clearInterval(this.interval);
        }

        if (this.anomalies.length === 0) {
            this.showErrorToast('No anomaly data available');
            return;
        }

        // Reset displayed counters
        this.displayedAnomalies = { high: 0, medium: 0, low: 0 };
        this.totalDisplayed = 0;
        this.updateLiveCounters();

        // Show first random toast immediately
        this.showRandomAnomaly();

        // Then show every 10 seconds
        this.interval = setInterval(() => {
            this.showRandomAnomaly();
        }, 10000);
    }

    stopToastInterval() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }

    showRandomAnomaly() {
        if (this.anomalies.length === 0) {
            return;
        }

        // Pick a random anomaly from the JSON data
        const randomIndex = Math.floor(Math.random() * this.anomalies.length);
        const anomaly = this.anomalies[randomIndex];
        const vehicleId = this.extractEquipmentId(anomaly);
        
        console.log(`🚨 Displaying anomaly for Vehicle ID: ${vehicleId} (Index: ${randomIndex}, Severity: ${anomaly.severity})`);
        
        // Update counters BEFORE showing the toast
        this.incrementDisplayedCounter(anomaly.severity);
        
        this.showAnomalyToast(anomaly);
    }

    incrementDisplayedCounter(severity) {
        const severityLower = severity.toLowerCase();
        if (severityLower === 'high') this.displayedAnomalies.high++;
        else if (severityLower === 'medium') this.displayedAnomalies.medium++;
        else if (severityLower === 'low') this.displayedAnomalies.low++;
        
        this.totalDisplayed++;
        this.updateLiveCounters();
    }

    updateLiveCounters() {
        // Update the live displayed counters on the dashboard
        const highElement = document.getElementById('liveHighCount');
        const mediumElement = document.getElementById('liveMediumCount');
        const lowElement = document.getElementById('liveLowCount');
        const totalElement = document.getElementById('liveTotalCount');

        if (highElement) {
            highElement.textContent = this.displayedAnomalies.high;
            this.animateCounter(highElement);
        }
        if (mediumElement) {
            mediumElement.textContent = this.displayedAnomalies.medium;
            this.animateCounter(mediumElement);
        }
        if (lowElement) {
            lowElement.textContent = this.displayedAnomalies.low;
            this.animateCounter(lowElement);
        }
        if (totalElement) {
            totalElement.textContent = this.totalDisplayed;
            this.animateCounter(totalElement);
        }

        // Update live chart if function exists
        if (window.updateLiveChart) {
            window.updateLiveChart(this.displayedAnomalies);
        }
    }

    animateCounter(element) {
        // Add a pulse animation to the counter
        element.classList.add('counter-updated');
        setTimeout(() => {
            element.classList.remove('counter-updated');
        }, 600);
    }

    resetLiveCounters() {
        this.displayedAnomalies = { high: 0, medium: 0, low: 0 };
        this.totalDisplayed = 0;
        this.updateLiveCounters();
    }

    showAnomalyToast(anomaly) {
        const toast = this.createToast(anomaly);
        this.container.appendChild(toast);

        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);

        // Auto-hide after 8 seconds
        setTimeout(() => this.hideToast(toast), 8000);

        // Limit number of visible toasts
        this.limitVisibleToasts();
    }

    createToast(anomaly) {
        const toast = document.createElement('div');
        toast.className = `toast severity-${anomaly.severity.toLowerCase()}`;

        const severityIcon = this.getSeverityIcon(anomaly.severity);
        const topFeatures = anomaly.top_deviant_features || [];
        const vehicleId = this.extractEquipmentId(anomaly);
        
        // Format the anomaly type for better readability
        const anomalyType = this.formatAnomalyType(anomaly.anomaly_type);
        
        toast.innerHTML = `
            <div class="toast-header">
                <div class="toast-title">
                    <span class="severity-indicator"></span>
                    ${severityIcon} Vehicle Alert - ${anomaly.severity}
                </div>
                <div class="toast-time">${this.formatTime(anomaly.timestamp)}</div>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
            <div class="toast-body">
                <strong>Vehicle ID:</strong> <span style="color: #007bff; font-weight: bold;">${vehicleId}</span><br>
                <strong>Anomaly Score:</strong> <span style="color: ${this.getScoreColor(anomaly.anomaly_score)}">${anomaly.anomaly_score.toFixed(3)}</span><br>
                <strong>Issue:</strong> ${anomalyType}<br>
                <strong>Data Cluster:</strong> ${anomaly.cluster_id}<br>
                ${topFeatures.length > 0 ? `
                <div class="anomaly-details">
                    <strong>Critical Sensors:</strong><br>
                    ${topFeatures.slice(0, 3).map(feature => 
                        `<span class="feature-deviation" style="background: ${this.getDeviationColor(feature.deviation_percentage)}">
                            ${this.formatFeatureName(feature.feature)}: ${feature.deviation_percentage > 0 ? '+' : ''}${feature.deviation_percentage.toFixed(1)}%
                        </span>`
                    ).join(' ')}
                </div>` : ''}
            </div>
            <div class="toast-progress" style="width: 100%"></div>
        `;

        // Add progress bar animation
        const progressBar = toast.querySelector('.toast-progress');
        setTimeout(() => {
            progressBar.style.width = '0%';
            progressBar.style.transition = 'width 8s linear';
        }, 100);

        return toast;
    }

    extractEquipmentId(anomaly) {
        // Generate realistic Caterpillar vehicle IDs based on anomaly data
        const catEquipmentTypes = [
            'CAT320', 'CAT336', 'CAT349', 'CAT773', 'CAT777',  // Excavators & Trucks
            'CAT950', 'CAT966', 'CAT980', 'CAT992',            // Wheel Loaders  
            'CAT825', 'CAT836', 'CAT844',                      // Compactors
            'CAT735', 'CAT745', 'CAT785'                       // Articulated Trucks
        ];
        
        // Use cluster_id to determine equipment type for consistency
        const typeIndex = anomaly.cluster_id % catEquipmentTypes.length;
        const equipmentModel = catEquipmentTypes[typeIndex];
        
        // Generate consistent vehicle ID based on index
        // Format: CAT320-V2024-1234 (Model-Year-SerialNumber)
        const serialNumber = String(1000 + (anomaly.index % 8999)).padStart(4, '0');
        const vehicleId = `${equipmentModel}-V2024-${serialNumber}`;
        
        return vehicleId;
    }

    formatAnomalyType(anomalyType) {
        // Clean up the anomaly type text for better readability
        return anomalyType
            .replace(/High-Value Anomaly \(/, '')
            .replace(/\)$/, '')
            .replace('Multiple features elevated', 'Multiple sensors elevated')
            .replace('max:', 'Peak deviation:')
            .substring(0, 80) + (anomalyType.length > 80 ? '...' : '');
    }

    formatFeatureName(feature) {
        // Convert snake_case to readable names
        return feature
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace('Temp', 'Temperature')
            .replace('Pressure', 'Press.');
    }

    getScoreColor(score) {
        if (score > 3.0) return '#dc3545'; // Red for high scores
        if (score > 2.5) return '#fd7e14'; // Orange for medium-high
        if (score > 2.0) return '#ffc107'; // Yellow for medium
        return '#28a745'; // Green for low scores
    }

    getDeviationColor(percentage) {
        const absPercentage = Math.abs(percentage);
        if (absPercentage > 150) return 'rgba(220, 53, 69, 0.2)'; // Red background
        if (absPercentage > 100) return 'rgba(255, 193, 7, 0.2)'; // Yellow background
        return 'rgba(23, 162, 184, 0.2)'; // Blue background
    }

    getSeverityIcon(severity) {
        switch (severity.toLowerCase()) {
            case 'high': return '🚨';
            case 'medium': return '⚠️';
            case 'low': return '⚡';
            default: return '📊';
        }
    }

    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleString('en-US', { 
                month: 'short',
                day: 'numeric',
                hour: '2-digit', 
                minute: '2-digit',
                hour12: true 
            });
        } catch (error) {
            return 'Unknown time';
        }
    }

    hideToast(toast) {
        toast.classList.add('removing');
        setTimeout(() => {
            if (toast.parentElement) {
                toast.parentElement.removeChild(toast);
            }
        }, 300);
    }

    limitVisibleToasts() {
        const toasts = this.container.querySelectorAll('.toast:not(.removing)');
        if (toasts.length > 5) {
            // Remove oldest toasts if more than 5 are visible
            for (let i = 0; i < toasts.length - 5; i++) {
                this.hideToast(toasts[i]);
            }
        }
    }

    showErrorToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast severity-high';
        
        toast.innerHTML = `
            <div class="toast-header">
                <div class="toast-title">
                    <span class="severity-indicator"></span>
                    🚨 System Error
                </div>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;

        this.container.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => this.hideToast(toast), 8000);
    }

    clearAllToasts() {
        const toasts = this.container.querySelectorAll('.toast');
        toasts.forEach(toast => this.hideToast(toast));
    }
}

// Global toast manager instance
let toastManager;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async function() {
    toastManager = new ToastManager();
    
    // Load anomalies and start showing random toasts
    const loaded = await toastManager.loadAnomalies();
    if (loaded) {
        toastManager.startToastInterval();
        console.log('Toast system initialized with real-time dashboard updates');
    }
});

// Enhanced control functions for the dashboard
function toggleToasts() {
    if (toastManager.interval) {
        toastManager.stopToastInterval();
        document.getElementById('toggleButton').textContent = 'Start Alerts';
        document.getElementById('toggleButton').classList.remove('btn-danger');
        document.getElementById('toggleButton').classList.add('btn-success');
    } else {
        toastManager.startToastInterval();
        document.getElementById('toggleButton').textContent = 'Stop Alerts';
        document.getElementById('toggleButton').classList.remove('btn-success');
        document.getElementById('toggleButton').classList.add('btn-danger');
    }
}

function clearToasts() {
    toastManager.clearAllToasts();
}

function resetCounters() {
    if (toastManager) {
        toastManager.resetLiveCounters();
    }
}

function showSampleToast() {
    if (toastManager && toastManager.anomalies.length > 0) {
        // Show a random real anomaly as sample
        toastManager.showRandomAnomaly();
    } else {
        // Fallback sample if no data loaded
        const sampleAnomaly = {
            index: 999,
            anomaly_score: 3.456,
            anomaly_type: "Sample High-Value Anomaly (Multiple features elevated, max: engine_temp +185.2%)",
            severity: "HIGH",
            cluster_id: 2,
            timestamp: new Date().toISOString(),
            top_deviant_features: [
                { feature: "engine_temp", deviation_percentage: 185.2 },
                { feature: "oil_pressure", deviation_percentage: -156.7 },
                { feature: "fuel_level", deviation_percentage: 234.1 }
            ]
        };
        toastManager.incrementDisplayedCounter(sampleAnomaly.severity);
        toastManager.showAnomalyToast(sampleAnomaly);
    }
}
