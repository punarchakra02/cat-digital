import qrcode
from io import BytesIO
from django.core.files import File
from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta

# Equipment Type Choices based on problem statement
EQUIPMENT_TYPES = [
    ('Excavator', 'Excavator'),
    ('Crane', 'Crane'),
    ('Bulldozer', 'Bulldozer'),
    ('Grader', 'Grader'),
    ('Loader', 'Loader'),
    ('Compactor', 'Compactor'),
    ('Dump Truck', 'Dump Truck'),
]

# Equipment Status
EQUIPMENT_STATUS = [
    ('Available', 'Available'),
    ('Rented', 'Rented'),
    ('Under Maintenance', 'Under Maintenance'),
    ('Out of Service', 'Out of Service'),
]

class Machine(models.Model):
    equipment_id = models.CharField(max_length=100, unique=True, help_text="Unique equipment identifier (e.g., EQX1001)")
    type = models.CharField(max_length=100, choices=EQUIPMENT_TYPES)
    manufacturer = models.CharField(max_length=100, default='Unknown')
    model = models.CharField(max_length=100, default='Unknown')
    year_manufactured = models.IntegerField(default=2020)
    status = models.CharField(max_length=50, choices=EQUIPMENT_STATUS, default='Available')
    list_price_per_day = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    location = models.CharField(max_length=200, blank=True, null=True, help_text="Current location/depot")
    qr_code = models.ImageField(upload_to="qr_codes", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['equipment_id']

    def __str__(self):
        return f"{self.equipment_id} - {self.type}"

    @property
    def is_available(self):
        return self.status == 'Available'

    @property
    def current_rental(self):
        return self.rentals.filter(active=True, end_date__gt=timezone.now()).first()

    @property
    def total_runtime_hours(self):
        """Calculate total runtime hours from all usage records"""
        total = self.usages.aggregate(models.Sum('engine_hours'))['engine_hours__sum'] or 0
        return total

    @property
    def total_idle_hours(self):
        """Calculate total idle hours from all usage records"""
        total = self.usages.aggregate(models.Sum('idle_hours'))['idle_hours__sum'] or 0
        return total

    def save(self, *args, **kwargs):
        # Generate QR code based on equipment_id
        if not self.qr_code:
            qr_img = qrcode.make(self.equipment_id)
            canvas = BytesIO()
            qr_img.save(canvas, format="PNG")
            file_name = f"qr_{self.equipment_id}.png"

            # Save to ImageField
            self.qr_code.save(file_name, File(canvas), save=False)
            canvas.close()

        super().save(*args, **kwargs)


# Rental Status
RENTAL_STATUS = [
    ('Active', 'Active'),
    ('Completed', 'Completed'),
    ('Overdue', 'Overdue'),
    ('Cancelled', 'Cancelled'),
]

class Rental(models.Model):
    rental_id = models.CharField(max_length=100, unique=True, blank=True)
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="rentals")
    operator_id = models.CharField(max_length=100, help_text="Operator/User ID")
    operator_name = models.CharField(max_length=200, default='Unknown')
    site_id = models.CharField(max_length=100, help_text="Site where equipment is deployed")
    site_name = models.CharField(max_length=200, blank=True, null=True)
    start_date = models.DateTimeField(auto_now_add=True)
    expected_end_date = models.DateTimeField()
    actual_end_date = models.DateTimeField(blank=True, null=True)
    active = models.BooleanField(default=True)
    status = models.CharField(max_length=50, choices=RENTAL_STATUS, default='Active')
    rental_cost_per_day = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-start_date']

    def __str__(self):
        return f"{self.machine.equipment_id} rented to {self.operator_id} at {self.site_id}"

    @property
    def is_overdue(self):
        """Check if rental is overdue"""
        return timezone.now() > self.expected_end_date and self.active

    @property
    def days_rented(self):
        """Calculate total days rented"""
        end_date = self.actual_end_date if self.actual_end_date else timezone.now()
        return (end_date - self.start_date).days

    @property
    def total_cost(self):
        """Calculate total rental cost"""
        return self.rental_cost_per_day * self.days_rented

    def save(self, *args, **kwargs):
        # Generate rental ID if not provided
        if not self.rental_id:
            self.rental_id = f"R{self.machine.equipment_id}_{timezone.now().strftime('%Y%m%d%H%M')}"
        
        # Update machine status
        if self.active:
            self.machine.status = 'Rented'
            self.machine.save()
        
        # Update status based on dates
        if self.is_overdue:
            self.status = 'Overdue'
        elif self.actual_end_date:
            self.status = 'Completed'
            self.active = False
            self.machine.status = 'Available'
            self.machine.save()
            
        super().save(*args, **kwargs)

    def return_equipment(self):
        """Mark equipment as returned"""
        self.actual_end_date = timezone.now()
        self.active = False
        self.status = 'Completed'
        self.machine.status = 'Available'
        self.save()


class EquipmentUsage(models.Model):
    """Daily usage tracking for equipment"""
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="usages")
    rental = models.ForeignKey(Rental, on_delete=models.CASCADE, related_name="usage_logs", blank=True, null=True)
    site_id = models.CharField(max_length=100)
    site_name = models.CharField(max_length=200, blank=True, null=True)
    operator_id = models.CharField(max_length=100)
    date = models.DateField(default=timezone.now)
    
    # Daily metrics
    engine_hours = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Engine runtime hours for the day")
    idle_hours = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Idle hours for the day")
    fuel_consumed = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Fuel consumed in liters")
    distance_traveled = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Distance in kilometers")
    
    # Location tracking
    start_location = models.CharField(max_length=200, blank=True, null=True)
    end_location = models.CharField(max_length=200, blank=True, null=True)
    
    # Performance metrics
    productivity_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.00, help_text="Daily productivity score (0-100)")
    maintenance_alerts = models.IntegerField(default=0, help_text="Number of maintenance alerts triggered")
    
    # Additional data
    weather_conditions = models.CharField(max_length=100, blank=True, null=True)
    work_type = models.CharField(max_length=200, blank=True, null=True, help_text="Type of work performed")
    notes = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']
        unique_together = ['machine', 'date']  # One usage record per machine per day

    def __str__(self):
        return f"{self.machine.equipment_id} usage on {self.date}"

    @property
    def efficiency_ratio(self):
        """Calculate efficiency as engine_hours / (engine_hours + idle_hours)"""
        total_hours = self.engine_hours + self.idle_hours
        if total_hours > 0:
            return (self.engine_hours / total_hours) * 100
        return 0

    @property
    def fuel_efficiency(self):
        """Calculate fuel efficiency as fuel consumed per engine hour"""
        if self.engine_hours > 0:
            return self.fuel_consumed / self.engine_hours
        return 0


# Component choices for equipment health monitoring
COMPONENT_CHOICES = [
    ('Engine', 'Engine'),
    ('Hydraulics', 'Hydraulics'),
    ('Transmission', 'Transmission'),
    ('Fuel System', 'Fuel System'),
    ('Cooling System', 'Cooling System'),
    ('Electrical', 'Electrical'),
    ('Brakes', 'Brakes'),
    ('MISC', 'MISC'),
]

# Health status choices
HEALTH_STATUS = [
    ('Good', 'Good'),
    ('Warning', 'Warning'),
    ('Critical', 'Critical'),
    ('Maintenance Required', 'Maintenance Required'),
]

class EquipmentHealth(models.Model):
    """Real-time health monitoring for equipment"""
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="health_records")
    component = models.CharField(max_length=50, choices=COMPONENT_CHOICES)
    status = models.CharField(max_length=50, choices=HEALTH_STATUS, default='Good')
    
    # Engine parameters
    engine_temperature = models.FloatField(null=True, blank=True, help_text="Engine temperature in Celsius")
    oil_pressure = models.FloatField(null=True, blank=True, help_text="Oil pressure in PSI")
    coolant_temperature = models.FloatField(null=True, blank=True, help_text="Coolant temperature in Celsius")
    air_filter_pressure = models.FloatField(null=True, blank=True, help_text="Air filter pressure differential")
    exhaust_gas_temperature = models.FloatField(null=True, blank=True, help_text="Exhaust gas temperature")
    
    # Hydraulic system
    hydraulic_pressure = models.FloatField(null=True, blank=True, help_text="Hydraulic system pressure")
    hydraulic_temperature = models.FloatField(null=True, blank=True, help_text="Hydraulic fluid temperature")
    hydraulic_pump_rate = models.FloatField(null=True, blank=True, help_text="Hydraulic pump flow rate")
    
    # Fuel system
    fuel_level = models.FloatField(null=True, blank=True, help_text="Fuel level percentage")
    fuel_consumption_rate = models.FloatField(null=True, blank=True, help_text="Fuel consumption rate L/hr")
    water_in_fuel = models.FloatField(null=True, blank=True, help_text="Water contamination level in fuel")
    
    # Electrical system
    battery_voltage = models.FloatField(null=True, blank=True, help_text="Battery voltage")
    system_voltage = models.FloatField(null=True, blank=True, help_text="System voltage")
    alternator_output = models.FloatField(null=True, blank=True, help_text="Alternator output")
    
    # Transmission and brakes
    transmission_temperature = models.FloatField(null=True, blank=True, help_text="Transmission temperature")
    transmission_pressure = models.FloatField(null=True, blank=True, help_text="Transmission pressure")
    brake_pressure = models.FloatField(null=True, blank=True, help_text="Brake system pressure")
    brake_control = models.FloatField(null=True, blank=True, help_text="Brake control response")
    
    # Sensor readings
    vibration_level = models.FloatField(null=True, blank=True, help_text="Vibration level")
    pedal_sensor = models.FloatField(null=True, blank=True, help_text="Pedal sensor reading")
    
    # Alerts and notifications
    alert_code = models.CharField(max_length=50, blank=True, null=True, help_text="Alert/Error code")
    alert_message = models.TextField(blank=True, null=True, help_text="Alert description")
    severity = models.IntegerField(default=1, help_text="Severity level 1-5")
    
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(blank=True, null=True)
    technician_notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Equipment Health Record"
        verbose_name_plural = "Equipment Health Records"

    def __str__(self):
        return f"{self.machine.equipment_id} - {self.component} - {self.status} ({self.timestamp.strftime('%Y-%m-%d %H:%M')})"

    @property
    def is_critical(self):
        """Check if this health record indicates a critical issue"""
        return self.status in ['Critical', 'Maintenance Required'] or self.severity >= 4

    @property
    def requires_attention(self):
        """Check if this health record requires attention"""
        return self.status in ['Warning', 'Critical', 'Maintenance Required'] and not self.resolved


# Predictive analytics model for demand forecasting
class DemandForecast(models.Model):
    """Predictive demand forecasting for equipment"""
    equipment_type = models.CharField(max_length=100, choices=EQUIPMENT_TYPES)
    site_id = models.CharField(max_length=100)
    site_name = models.CharField(max_length=200, blank=True, null=True)
    forecast_date = models.DateField()
    predicted_demand = models.IntegerField(help_text="Predicted number of units needed")
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, help_text="Prediction confidence (0-100)")
    
    # Factors influencing demand
    historical_usage = models.IntegerField(default=0)
    seasonal_factor = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)
    project_requirements = models.IntegerField(default=0)
    weather_impact = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['forecast_date']
        unique_together = ['equipment_type', 'site_id', 'forecast_date']

    def __str__(self):
        return f"{self.equipment_type} demand forecast for {self.site_id} on {self.forecast_date}"


# Anomaly detection model
class AnomalyAlert(models.Model):
    """Anomaly detection alerts"""
    ANOMALY_TYPES = [
        ('Idle_Time', 'Excessive Idle Time'),
        ('Fuel_Consumption', 'Abnormal Fuel Consumption'),
        ('Unassigned_Equipment', 'Unassigned Equipment'),
        ('Location_Drift', 'Unexpected Location Change'),
        ('Usage_Pattern', 'Unusual Usage Pattern'),
        ('Performance_Drop', 'Performance Degradation'),
    ]
    
    SEVERITY_LEVELS = [
        (1, 'Low'),
        (2, 'Medium'),
        (3, 'High'),
        (4, 'Critical'),
    ]
    
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="anomalies")
    anomaly_type = models.CharField(max_length=50, choices=ANOMALY_TYPES)
    severity = models.IntegerField(choices=SEVERITY_LEVELS, default=2)
    description = models.TextField()
    detected_at = models.DateTimeField(auto_now_add=True)
    
    # Anomaly details
    threshold_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    actual_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, help_text="Detection confidence (0-100)")
    
    # Resolution tracking
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.CharField(max_length=100, blank=True, null=True)
    acknowledged_at = models.DateTimeField(blank=True, null=True)
    resolved = models.BooleanField(default=False)
    resolution_notes = models.TextField(blank=True, null=True)
    resolved_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ['-detected_at']

    def __str__(self):
        return f"{self.machine.equipment_id} - {self.get_anomaly_type_display()} (Severity: {self.get_severity_display()})"