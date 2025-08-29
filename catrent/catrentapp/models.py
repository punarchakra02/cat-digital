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
    ('Loader', 'Loader'),
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
        checkout_url = f"https://catrent.onrender.com/checkout/{self.equipment_id}/"

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

    # ----------------------
    # Foreign key + identifiers
    # ----------------------
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='rentals')
    rental_id = models.CharField(max_length=100, unique=True, blank=True)
    operator_id = models.CharField(max_length=100, help_text="Operator/User ID")
    operator_name = models.CharField(max_length=200, default="Unknown")
    site_id = models.CharField(max_length=100, help_text="Site where equipment is deployed")
    site_name = models.CharField(max_length=200, blank=True, null=True)

    # ----------------------
    # Rental lifecycle
    # ----------------------
    start_date = models.DateTimeField()   # checkout_date
    expected_end_date = models.DateTimeField()             # planned checkin
    actual_end_date = models.DateTimeField(blank=True, null=True)
    active = models.BooleanField(default=True)
    status = models.CharField(max_length=50, choices=RENTAL_STATUS_CHOICES, default='Active')
    notes = models.TextField(blank=True, null=True)

    # ----------------------
    # Primary features
    # ----------------------
    rate_per_day = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    week = models.CharField(max_length=10, help_text="Year-week (e.g., 2025-W35)", default=datetime.now().strftime("%Y-W%U"))
    maintenance = models.IntegerField(default=0)
    engine_hours = models.IntegerField(default=0)
    idle_hours = models.IntegerField(default=0)
    rental_duration = models.IntegerField(default=0)   # days
    usage_efficiency = models.FloatField(default=0.0)
    engine_hours_per_day = models.FloatField(default=0.0)
    maintenance_per_day = models.FloatField(default=0.0)

    # ----------------------
    # Derived demand features
    # ----------------------
    target_checkout_count = models.IntegerField(default=0)
    checkout_count_t_1 = models.IntegerField(blank=True, null=True)
    checkout_count_t_4 = models.IntegerField(blank=True, null=True)
    rolling_mean_4w = models.FloatField(blank=True, null=True)
    rolling_std_4w = models.FloatField(blank=True, null=True)
    rolling_max_4w = models.FloatField(blank=True, null=True)

    # ----------------------
    # Pricing features
    # ----------------------
    price_index = models.FloatField(default=1.0)
    rate_relative = models.FloatField(default=1.0)

    # ----------------------
    # Maintenance features
    # ----------------------
    pct_maint = models.FloatField(default=0.0)
    recent_maintenance = models.BooleanField(default=False)

    # ----------------------
    # Equipment-type features
    # ----------------------
    equipment_type_encoded = models.IntegerField(default=0)

    # ----------------------
    # Engine/idle aggregations
    # ----------------------
    engine_hours_avg = models.FloatField(default=0.0)
    idle_ratio_avg = models.FloatField(default=0.0)

    # ----------------------
    # Seasonal encodings
    # ----------------------
    month_sin = models.FloatField(default=0.0)
    month_cos = models.FloatField(default=0.0)
    week_sin = models.FloatField(default=0.0)
    week_cos = models.FloatField(default=0.0)

    # ----------------------
    # Metadata
    # ----------------------
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-start_date']

    def _str_(self):
        return f"Rental {self.rental_id} - {self.machine.equipment_id}"

    def save(self, *args, **kwargs):
        # Generate rental_id if it doesn't exist
        if not self.rental_id:
            from django.utils.timezone import now
            timestamp = now().strftime('%Y%m%d%H%M%S')
            self.rental_id = f"RNT{timestamp}{self.machine.equipment_id[-3:]}"
        
        # Calculate rental_duration from dates
        if self.start_date and self.expected_end_date:
            # Ensure both datetimes are timezone-aware
            from django.utils.timezone import make_aware, is_aware
            start = self.start_date
            end = self.expected_end_date
            if not is_aware(start):
                start = make_aware(start)
            if not is_aware(end):
                end = make_aware(end)
            # Calculate duration in days
            duration = (end - start).days
            self.rental_duration = max(duration, 1)  # Ensure at least 1 day
        
        # Calculate derived features based on primary features
        if self.engine_hours and self.idle_hours and self.rental_duration:
            # Calculate usage efficiency
            self.usage_efficiency = (self.engine_hours - self.idle_hours) / max(self.engine_hours, 1)
            
            # Calculate engine hours per day
            self.engine_hours_per_day = self.engine_hours / max(self.rental_duration, 1)
            
            # Calculate maintenance per day
            self.maintenance_per_day = self.maintenance / max(self.rental_duration, 1)
            
            # Set recent_maintenance flag
            self.recent_maintenance = self.maintenance > 0
            
        # Calculate seasonal encodings if start_date is set
        if self.start_date:
            import numpy as np
            # Month-based seasonality
            month = self.start_date.month
            self.month_sin = np.sin(2 * np.pi * month / 12)
            self.month_cos = np.cos(2 * np.pi * month / 12)
            
            # Week-based seasonality
            week_num = self.start_date.isocalendar()[1]  # ISO week number
            self.week_sin = np.sin(2 * np.pi * week_num / 52)
            self.week_cos = np.cos(2 * np.pi * week_num / 52)
        
        # Populate the week field with the ISO year-week format
        if self.start_date and not self.week:
            year = self.start_date.year
            week_num = self.start_date.isocalendar()[1]
            self.week = f"{year}-W{week_num:02d}"
        
        super().save(*args, **kwargs)
        
        # After saving, calculate and update additional derived features that need other rentals
        self.calculate_aggregated_features()

    def calculate_aggregated_features(self):
        """Calculate features that require aggregations across multiple rentals"""
        from django.db.models import Avg, Count, Max, StdDev
        import pandas as pd
        
        # Get equipment type from machine
        equipment_type = self.machine.type
        
        # Get all rentals for this equipment type
        type_rentals = Rental.objects.filter(machine__type=equipment_type)
        
        # Calculate target_checkout_count (number of rentals for this equipment type in the same week)
        week_rentals = type_rentals.filter(week=self.week)
        self.target_checkout_count = week_rentals.count()
        
        if self.target_checkout_count > 0:
            # Calculate rate_relative
            avg_rate = type_rentals.aggregate(Avg('rate_per_day'))['rate_per_day__avg'] or 1.0
            self.rate_relative = float(self.rate_per_day) / float(avg_rate)
            
            # Calculate price_index
            self.price_index = float(self.rate_per_day) / float(avg_rate)
            
            # Calculate pct_maint
            self.pct_maint = self.maintenance / self.target_checkout_count
            
            # Calculate engine_hours_avg
            self.engine_hours_avg = self.engine_hours / self.target_checkout_count
            
            # Calculate idle_ratio_avg
            total_hours = self.engine_hours + self.idle_hours
            self.idle_ratio_avg = self.idle_hours / max(total_hours, 1)
        
        # Get rental counts for previous weeks
        if self.start_date:
            from datetime import timedelta
            
            # Calculate equipment_type_encoded
            equipment_types = list(set(Machine.objects.values_list('type', flat=True)))
            if equipment_types:
                self.equipment_type_encoded = equipment_types.index(equipment_type) if equipment_type in equipment_types else 0
            
            # For checkout_count_t_1 and checkout_count_t_4, we need to check rentals from previous weeks
            # This is simplified and might need adjustment based on your exact requirements
            one_week_ago = self.start_date - timedelta(days=7)
            four_weeks_ago = self.start_date - timedelta(days=28)
            
            # Get the ISO week for 1 and 4 weeks ago
            one_week_ago_iso = one_week_ago.isocalendar()
            four_weeks_ago_iso = four_weeks_ago.isocalendar()
            
            week_t1 = f"{one_week_ago_iso[0]}-W{one_week_ago_iso[1]:02d}"
            week_t4 = f"{four_weeks_ago_iso[0]}-W{four_weeks_ago_iso[1]:02d}"
            
            # Count rentals in previous weeks
            count_t1 = type_rentals.filter(week=week_t1).count()
            count_t4 = type_rentals.filter(week=week_t4).count()
            
            self.checkout_count_t_1 = count_t1
            self.checkout_count_t_4 = count_t4
            
            # Calculate rolling mean, std, and max for 4 weeks
            # This is a simplified approach
            weeks_to_check = [week_t4]
            for i in range(1, 4):
                check_date = four_weeks_ago + timedelta(days=i*7)
                check_week = f"{check_date.isocalendar()[0]}-W{check_date.isocalendar()[1]:02d}"
                weeks_to_check.append(check_week)
            
            rolling_counts = [type_rentals.filter(week=w).count() for w in weeks_to_check]
            
            if rolling_counts:
                import numpy as np
                self.rolling_mean_4w = float(np.mean(rolling_counts))
                self.rolling_std_4w = float(np.std(rolling_counts))
                self.rolling_max_4w = float(np.max(rolling_counts))
        
        # Save the updated derived features
        Rental.objects.filter(pk=self.pk).update(
            target_checkout_count=self.target_checkout_count,
            rate_relative=self.rate_relative,
            price_index=self.price_index,
            pct_maint=self.pct_maint,
            engine_hours_avg=self.engine_hours_avg,
            idle_ratio_avg=self.idle_ratio_avg,
            equipment_type_encoded=self.equipment_type_encoded,
            checkout_count_t_1=self.checkout_count_t_1,
            checkout_count_t_4=self.checkout_count_t_4,
            rolling_mean_4w=self.rolling_mean_4w,
            rolling_std_4w=self.rolling_std_4w,
            rolling_max_4w=self.rolling_max_4w
        )

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

class Operator(models.Model):
    operator_id = models.CharField(max_length=20, unique=True)  # auto-generated
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

    def __str__(self):
        return f"{self.operator_id} - {self.name}"
