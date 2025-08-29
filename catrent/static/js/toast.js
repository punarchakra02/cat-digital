// Toast notification system for anomaly alerts
class ToastNotification {
    constructor() {
        this.toastContainer = null;
        this.createToastContainer();
        this.anomalyData = [];
        this.severityCounts = {
            'Critical': 0,
            'Warning': 0,
            'Medium': 0,
            'Low': 0
        };
        this.init();
    }

    createToastContainer() {
        // Create toast container if it doesn't exist
        this.toastContainer = document.getElementById('toast-container');
        if (!this.toastContainer) {
            this.toastContainer = document.createElement('div');
            this.toastContainer.id = 'toast-container';
            this.toastContainer.className = 'toast-container';
            document.body.appendChild(this.toastContainer);
        }
    }

    async loadAnomalyData() {
        try {
            const response = await fetch('/static/detailed_anomaly_report.json');
            if (!response.ok) {
                throw new Error('Failed to load anomaly data');
            }
            this.anomalyData = await response.json();
            this.updateSeverityCounts();
            console.log('Anomaly data loaded:', this.anomalyData.length, 'records');
        } catch (error) {
            console.error('Error loading anomaly data:', error);
            // Fallback to sample data
            this.anomalyData = this.getSampleData();
            this.updateSeverityCounts();
        }
    }

    getSampleData() {
        return [
            {
                "vehicle_id": "BU3847",
                "equipment_type": "Bulldozer",
                "anomaly_type": "Engine Temperature",
                "severity": "Critical",
                "timestamp": new Date().toISOString(),
                "description": "Engine temperature exceeding normal operating range",
                "location": "Site A - Construction Zone",
                "operator": "John Smith",
                "maintenance_required": true
            },
            {
                "vehicle_id": "EX2156",
                "equipment_type": "Excavator",
                "anomaly_type": "Hydraulic Pressure",
                "severity": "Warning",
                "timestamp": new Date().toISOString(),
                "description": "Hydraulic pressure fluctuation detected",
                "location": "Site B - Excavation Area",
                "operator": "Sarah Johnson",
                "maintenance_required": false
            }
        ];
    }

    updateSeverityCounts() {
        // Reset counts
        Object.keys(this.severityCounts).forEach(key => {
            this.severityCounts[key] = 0;
        });

        // Count anomalies by severity
        this.anomalyData.forEach(anomaly => {
            if (this.severityCounts.hasOwnProperty(anomaly.severity)) {
                this.severityCounts[anomaly.severity]++;
            }
        });

        // Update counter displays
        this.updateCounterDisplays();
    }

    updateCounterDisplays() {
        Object.keys(this.severityCounts).forEach(severity => {
            const counterElement = document.getElementById(`${severity.toLowerCase()}-count`);
            if (counterElement) {
                counterElement.textContent = this.severityCounts[severity];
            }
        });
    }

    getRandomAnomaly() {
        if (this.anomalyData.length === 0) return null;
        const randomIndex = Math.floor(Math.random() * this.anomalyData.length);
        return this.anomalyData[randomIndex];
    }

    getSeverityClass(severity) {
        const severityClasses = {
            'Critical': 'toast-critical',
            'Warning': 'toast-warning', 
            'Medium': 'toast-medium',
            'Low': 'toast-low'
        };
        return severityClasses[severity] || 'toast-info';
    }

    getSeverityIcon(severity) {
        const severityIcons = {
            'Critical': '🚨',
            'Warning': '⚠️',
            'Medium': '🔶',
            'Low': 'ℹ️'
        };
        return severityIcons[severity] || '📋';
    }

    showToast(anomaly) {
        if (!anomaly) return;

        const toast = document.createElement('div');
        toast.className = `toast ${this.getSeverityClass(anomaly.severity)}`;
        
        const icon = this.getSeverityIcon(anomaly.severity);
        const timestamp = new Date(anomaly.timestamp).toLocaleTimeString();
        
        toast.innerHTML = `
            <div class="toast-header">
                <span class="toast-icon">${icon}</span>
                <strong class="toast-title">${anomaly.severity} Alert</strong>
                <small class="toast-time">${timestamp}</small>
                <button type="button" class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="toast-body">
                <div class="anomaly-info">
                    <strong>Vehicle:</strong> ${anomaly.vehicle_id} (${anomaly.equipment_type})
                </div>
                <div class="anomaly-info">
                    <strong>Issue:</strong> ${anomaly.anomaly_type}
                </div>
                <div class="anomaly-description">
                    ${anomaly.description}
                </div>
                <div class="anomaly-location">
                    📍 ${anomaly.location}
                </div>
                <div class="anomaly-operator">
                    👤 Operator: ${anomaly.operator}
                </div>
                ${anomaly.maintenance_required ? '<div class="maintenance-badge">🔧 Maintenance Required</div>' : ''}
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Auto-remove toast after 8 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => {
                    if (toast.parentNode) {
                        toast.remove();
                    }
                }, 300);
            }
        }, 8000);

        // Add slide-in animation
        toast.style.animation = 'slideIn 0.3s ease-out';
    }

    startNotifications() {
        // Show initial toast
        const randomAnomaly = this.getRandomAnomaly();
        if (randomAnomaly) {
            this.showToast(randomAnomaly);
        }

        // Set interval to show toasts every 10 seconds
        setInterval(() => {
            const randomAnomaly = this.getRandomAnomaly();
            if (randomAnomaly) {
                this.showToast(randomAnomaly);
            }
        }, 10000);
    }

    async init() {
        await this.loadAnomalyData();
        this.startNotifications();
    }
}

// CSS styles for toasts (inject into head)
function injectToastStyles() {
    const styles = `
        <style>
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
        }

        .toast {
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 10px;
            overflow: hidden;
            border-left: 4px solid #007bff;
            max-width: 100%;
            word-wrap: break-word;
        }

        .toast-critical {
            border-left-color: #dc3545;
        }

        .toast-warning {
            border-left-color: #ffc107;
        }

        .toast-medium {
            border-left-color: #fd7e14;
        }

        .toast-low {
            border-left-color: #17a2b8;
        }

        .toast-header {
            background: #f8f9fa;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #dee2e6;
        }

        .toast-icon {
            font-size: 16px;
            margin-right: 8px;
        }

        .toast-title {
            color: #495057;
            font-size: 14px;
            flex-grow: 1;
        }

        .toast-time {
            color: #6c757d;
            font-size: 12px;
            margin-left: 8px;
        }

        .toast-close {
            background: none;
            border: none;
            font-size: 18px;
            color: #6c757d;
            cursor: pointer;
            margin-left: 8px;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .toast-close:hover {
            color: #495057;
        }

        .toast-body {
            padding: 12px;
        }

        .anomaly-info {
            margin-bottom: 6px;
            font-size: 13px;
        }

        .anomaly-description {
            color: #495057;
            font-size: 13px;
            margin-bottom: 8px;
            font-style: italic;
        }

        .anomaly-location, .anomaly-operator {
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 4px;
        }

        .maintenance-badge {
            background: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 8px;
            display: inline-block;
        }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }

        @media (max-width: 480px) {
            .toast-container {
                left: 10px;
                right: 10px;
                top: 10px;
                max-width: none;
            }
        }
        </style>
    `;
    document.head.insertAdjacentHTML('beforeend', styles);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    injectToastStyles();
    // Small delay to ensure styles are applied
    setTimeout(() => {
        window.toastNotification = new ToastNotification();
    }, 100);
});