import qrcode
from io import BytesIO
from django.core.files import File
from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta

# Equipment Type Choices
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
    status = models.CharField(max_length=50, choices=EQUIPMENT_STATUS, default='Available')
    qr_code = models.ImageField(upload_to="qr_codes", blank=True, null=True)
    rate_per_day = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, help_text="Rental rate per day")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['equipment_id']

    def _str_(self):
        return f"{self.equipment_id} - {self.type}"

    @property
    def is_available(self):
        return self.status == 'Available'

    @property
    def current_rental(self):
        return self.rentals.filter(active=True).first()

    def save(self, *args, **kwargs):
        # Generate QR code if it doesn't exist
        if not self.qr_code:
            self.generate_qr_code()
        super().save(*args, **kwargs)

    def generate_qr_code(self):
        # Create QR code with checkout URL for admins
        checkout_url = f"https://catrent.vercel.app/checkout/{self.equipment_id}/"
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(checkout_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        filename = f"qr_{self.equipment_id}.png"
        self.qr_code.save(filename, File(buffer), save=False)


class Rental(models.Model):
    RENTAL_STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Completed', 'Completed'),
        ('Overdue', 'Overdue'),
        ('Cancelled', 'Cancelled'),
    ]

    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='rentals')
    rental_id = models.CharField(max_length=100, unique=True, blank=True)
    operator_id = models.CharField(max_length=100, help_text="Operator/User ID")
    operator_name = models.CharField(max_length=200, default="Unknown")
    site_id = models.CharField(max_length=100, help_text="Site where equipment is deployed")
    site_name = models.CharField(max_length=200, blank=True, null=True)
    start_date = models.DateTimeField(auto_now_add=True)
    expected_end_date = models.DateTimeField()
    actual_end_date = models.DateTimeField(blank=True, null=True)
    active = models.BooleanField(default=True)
    status = models.CharField(max_length=50, choices=RENTAL_STATUS_CHOICES, default='Active')
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-start_date']

    def _str_(self):
        return f"Rental {self.rental_id} - {self.machine.equipment_id}"

    def save(self, *args, **kwargs):
        if not self.rental_id:
            # Generate unique rental ID
            from django.utils.timezone import now
            timestamp = now().strftime('%Y%m%d%H%M%S')
            self.rental_id = f"RNT{timestamp}{self.machine.equipment_id[-3:]}"
        super().save(*args, **kwargs)

    @property
    def is_overdue(self):
        if self.active and self.expected_end_date:
            return timezone.now() > self.expected_end_date
        return False

    @property
    def duration_days(self):
        end_date = self.actual_end_date or timezone.now()
        return (end_date - self.start_date).days


class EquipmentUsage(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='usages')
    rental = models.ForeignKey(Rental, on_delete=models.CASCADE, related_name='usage_logs', blank=True, null=True)
    site_id = models.CharField(max_length=100)
    site_name = models.CharField(max_length=200, blank=True, null=True)
    operator_id = models.CharField(max_length=100)
    date = models.DateField(default=timezone.now)
    engine_hours = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Engine runtime hours for the day")
    idle_hours = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Idle hours for the day")
    fuel_consumed = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Fuel consumed in liters")
    distance_traveled = models.DecimalField(max_digits=8, decimal_places=2, default=0.00, help_text="Distance in kilometers")
    productivity_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.00, help_text="Daily productivity score (0-100)")
    work_type = models.CharField(max_length=200, blank=True, null=True, help_text="Type of work performed")
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']
        unique_together = ('machine', 'date')

    def _str_(self):
        return f"{self.machine.equipment_id} usage on {self.date}"


class EquipmentHealth(models.Model):
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

    HEALTH_STATUS_CHOICES = [
        ('Good', 'Good'),
        ('Warning', 'Warning'),
        ('Critical', 'Critical'),
        ('Maintenance Required', 'Maintenance Required'),
    ]

    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='health_records')
    component = models.CharField(max_length=50, choices=COMPONENT_CHOICES)
    status = models.CharField(max_length=50, choices=HEALTH_STATUS_CHOICES, default='Good')
    
    # Engine metrics
    engine_temperature = models.FloatField(blank=True, null=True, help_text="Engine temperature in Celsius")
    oil_pressure = models.FloatField(blank=True, null=True, help_text="Oil pressure in PSI")
    
    # Fuel metrics
    fuel_level = models.FloatField(blank=True, null=True, help_text="Fuel level percentage")
    fuel_consumption_rate = models.FloatField(blank=True, null=True, help_text="Fuel consumption rate L/hr")
    
    # Electrical metrics
    battery_voltage = models.FloatField(blank=True, null=True, help_text="Battery voltage")
    
    # General metrics
    alert_message = models.TextField(blank=True, null=True, help_text="Alert description")
    severity = models.IntegerField(default=1, help_text="Severity level 1-5")
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    technician_notes = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name = 'Equipment Health Record'
        verbose_name_plural = 'Equipment Health Records'
        ordering = ['-timestamp']

    def _str_(self):
        return f"{self.machine.equipment_id} - {self.component} ({self.status})"
