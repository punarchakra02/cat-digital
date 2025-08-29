from django.contrib import admin
from .models import Machine, Rental, EquipmentUsage, EquipmentHealth, Operator

# -----------------------------
# Machine Admin
# -----------------------------
@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ('equipment_id', 'type', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'type')
    search_fields = ('equipment_id', 'type')
    readonly_fields = ('qr_code', 'created_at', 'updated_at')

# -----------------------------
# Rental Admin
# -----------------------------
@admin.register(Rental)
class RentalAdmin(admin.ModelAdmin):
    list_display = ('rental_id', 'machine', 'operator_name', 'site_name', 'start_date', 'expected_end_date', 'actual_end_date', 'status', 'active')
    list_filter = ('status', 'active', 'start_date')
    search_fields = ('rental_id', 'machine__equipment_id', 'operator_name', 'site_name')
    readonly_fields = ('rental_id', 'start_date', 'created_at', 'updated_at')

# -----------------------------
# Equipment Usage Admin
# -----------------------------
@admin.register(EquipmentUsage)
class EquipmentUsageAdmin(admin.ModelAdmin):
    list_display = ('machine', 'rental', 'date', 'engine_hours', 'idle_hours', 'fuel_consumed', 'distance_traveled', 'productivity_score')
    list_filter = ('date', 'machine')
    search_fields = ('machine__equipment_id', 'rental__rental_id', 'operator_id', 'site_id')

# -----------------------------
# Equipment Health Admin
# -----------------------------
@admin.register(EquipmentHealth)
class EquipmentHealthAdmin(admin.ModelAdmin):
    list_display = ('machine', 'component', 'status', 'engine_temperature', 'fuel_level', 'battery_voltage', 'severity', 'timestamp', 'resolved')
    list_filter = ('status', 'component', 'resolved')
    search_fields = ('machine__equipment_id', 'component', 'alert_message')

@admin.register(Operator)
class OperatorAdmin(admin.ModelAdmin):
    list_display = ('operator_id', 'name', 'email')
