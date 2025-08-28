import qrcode
from io import BytesIO
from django.core.files import File
from django.db import models

class Machine(models.Model):
    equipment_id = models.CharField(max_length=100, unique=True)
    type = models.CharField(max_length=100)
    qr_code = models.ImageField(upload_to="qr_codes", blank=True, null=True)

    def __str__(self):
        return f"{self.equipment_id} - {self.type}"

    def save(self, *args, **kwargs):
        # Generate QR code based on equipment_id
        qr_img = qrcode.make(self.equipment_id)
        canvas = BytesIO()
        qr_img.save(canvas, format="PNG")
        file_name = f"qr_{self.equipment_id}.png"

        # Save to ImageField
        self.qr_code.save(file_name, File(canvas), save=False)
        canvas.close()

        super().save(*args, **kwargs)



class Rental(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="rentals")
    user_id = models.CharField(max_length=100)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField()
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.machine.equipment_id} rented to {self.user_id}"



class EquipmentUsage(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="usages")
    site_id = models.CharField(max_length=100)
    equipment_type = models.CharField(max_length=100)
    week = models.IntegerField()
    checkout_date = models.DateField()
    checkin_date = models.DateField()
    list_price = models.DecimalField(max_digits=10, decimal_places=2)
    maintenance_count = models.IntegerField()
    engine_hours = models.IntegerField()
    idle_hours = models.IntegerField()


COMPONENTS = [
    ("Engine", "Engine"),
    ("Fuel", "Fuel"),
    ("Drive", "Drive"),
    ("MISC", "MISC"),
]


class EquipmentHealth(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="health")
    component = models.CharField(max_length=50, choices=COMPONENTS)
    
    air_filter_pressure = models.FloatField(null=True, blank=True)
    exhaust_gas_temperature = models.FloatField(null=True, blank=True)
    hydrolic_pump_rate = models.FloatField(null=True, blank=True)
    oil_pressure = models.FloatField(null=True, blank=True)
    padel_sensor = models.FloatField(null=True, blank=True)
    system_voltage = models.FloatField(null=True, blank=True)
    water_in_fuel = models.FloatField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    brake_control = models.FloatField(null=True, blank=True)
    transmission_pressure = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)