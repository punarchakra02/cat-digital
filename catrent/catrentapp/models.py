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
