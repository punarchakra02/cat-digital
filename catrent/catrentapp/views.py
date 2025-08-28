from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse,JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse
from datetime import timedelta
import json
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .models import Machine, Rental, EquipmentHealth
from .forms import MachineForm, CheckoutForm

def add(request):
    """Add and list machines"""
    machines = Machine.objects.all()
    if request.method == "POST":
        form = MachineForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, 'Machine added successfully!')
                return redirect("add")
            except Exception as e:
                messages.error(request, 'Error adding machine.')
    else:
        form = MachineForm()
    return render(request, "add_machine.html", {"machines": machines, "form": form})


def delete_machine(request, pk):
    """Delete a machine"""
    try:
        machine = get_object_or_404(Machine, pk=pk)
        machine.delete()
        messages.success(request, 'Machine deleted successfully!')
    except Exception as e:
        messages.error(request, 'Error deleting machine.')
    return redirect("add")


def download_qr(request, pk):
    """Download QR code image"""
    machine = get_object_or_404(Machine, pk=pk)
    if not machine.qr_code:
        return HttpResponse("No QR code available for this equipment.")
    file_path = machine.qr_code.path
    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="image/png")
        response["Content-Disposition"] = f'attachment; filename="qr_{machine.equipment_id}.png"'
        return response


def rental_dashboard(request):
    """Dashboard showing rented, available machines and health chart"""
    try:
        # Rented machines
        rented_qs = Rental.objects.filter(active=True, status='Active').select_related('machine')
        rented = []
        for rental in rented_qs:
            latest_health = rental.machine.health_records.order_by('-timestamp').first()
            rented.append({
                "rental": rental,
                "health": latest_health,
            })

        # Available machines
        available = Machine.objects.filter(status='Available')

        # Latest health per machine (cross-db safe)
        latest_health_records = []
        for machine in Machine.objects.all():
            latest = machine.health_records.order_by('-timestamp').first()
            if latest:
                latest_health_records.append(latest)

        # Health chart data
        health_counts = {'Good':0, 'Warning':0, 'Critical':0, 'Maintenance Required':0}
        for h in latest_health_records:
            if h.status in health_counts:
                health_counts[h.status] += 1
        health_chart_data = {
            "labels": list(health_counts.keys()),
            "data": list(health_counts.values())
        }

        return render(request, "rental_dashboard.html", {
            "rented": rented,
            "available": available,
            "health_chart_data": json.dumps(health_chart_data),
        })

    except Exception as e:
        messages.error(request, 'Error loading dashboard.')
        return render(request, "rental_dashboard.html", {
            "rented": [],
            "available": [],
            "health_chart_data": json.dumps({"labels":[], "data":[]})
        })


def rent_machine(request, machine_id):
    """Rent a machine manually"""
    machine = get_object_or_404(Machine, equipment_id=machine_id)
    if machine.status == 'Rented':
        messages.error(request, 'Machine already rented!')
        return redirect("rental_dashboard")

    if request.method == "POST":
        operator_id = request.POST.get("operator_id")
        operator_name = request.POST.get("operator_name", "Unknown")
        site_id = request.POST.get("site_id")
        site_name = request.POST.get("site_name", "")
        days = int(request.POST.get("days", 1))
        expected_end_date = timezone.now() + timedelta(days=days)

        rental = Rental.objects.create(
            machine=machine,
            operator_id=operator_id,
            operator_name=operator_name,
            site_id=site_id,
            site_name=site_name,
            expected_end_date=expected_end_date,
            active=True,
            status='Active'
        )

        machine.status = 'Rented'
        machine.save()

        messages.success(request, f'Machine {machine.equipment_id} rented successfully!')
        return redirect("rental_dashboard")

    return render(request, "rent_machine.html", {"machine": machine})


def qr_scan_info(request, equipment_id):
    """QR scan checkout for admins"""
    machine = get_object_or_404(Machine, equipment_id=equipment_id)
    current_rental = machine.current_rental
    if current_rental:
        return render(request, 'checkout.html', {
            'machine': machine,
            'current_rental': current_rental,
            'is_already_rented': True,
        })

    if request.method == "POST":
        form = CheckoutForm(request.POST)
        if form.is_valid():
            rental = Rental.objects.create(
                machine=machine,
                operator_id=form.cleaned_data['operator_id'],
                operator_name=form.cleaned_data['operator_name'],
                site_id=form.cleaned_data['site_id'],
                site_name=form.cleaned_data['site_name'],
                start_date=form.cleaned_data['checkout_date'],
                expected_end_date=form.cleaned_data['expected_return_date'],
                active=True,
                status='Active'
            )
            machine.status = 'Rented'
            machine.save()
            messages.success(request, f'Equipment {machine.equipment_id} checked out!')
            return redirect('rental_dashboard')
    else:
        initial_data = {
            'equipment_id': machine.equipment_id,
            'type': machine.type,
            'status': machine.status,
            'checkout_date': timezone.now().strftime('%Y-%m-%dT%H:%M'),
        }
        form = CheckoutForm(initial=initial_data)

    return render(request, 'checkout.html', {
        'machine': machine,
        'form': form,
        'is_already_rented': False,
    })


@csrf_exempt  # We'll handle CSRF in JS fetch headers
def checkin_machine(request, rental_id):
    """
    Check-in a rented machine after QR validation.
    Expects POST with optional equipment_id for double validation.
    """
    rental = get_object_or_404(Rental, id=rental_id)

    if request.method == "POST":
        # Get equipment ID from POST for validation
        equipment_id = request.POST.get("equipment_id", None)
        if equipment_id and equipment_id != rental.machine.equipment_id:
            return JsonResponse({
                "status": "error",
                "message": "Equipment ID does not match. Check-in failed!"
            }, status=400)

        rental.active = False
        rental.status = 'Completed'
        rental.actual_end_date = timezone.now()
        rental.machine.status = 'Available'
        rental.machine.save()
        rental.save()

        return JsonResponse({
            "status": "success",
            "message": f"Machine {rental.machine.equipment_id} checked in successfully!"
        })

    return JsonResponse({"status": "error", "message": "Invalid request."}, status=400)


def anomaly_dashboard(request):
    """Display the anomaly monitoring dashboard"""
    return render(request, "anomaly_dashboard.html")


def anomaly_data(request):
    """API endpoint to serve anomaly data as JSON"""
    try:
        # Path to the detailed anomaly report
        anomaly_file_path = os.path.join(settings.BASE_DIR, 'detailed_anomaly_report.json')
        
        # Check if file exists
        if not os.path.exists(anomaly_file_path):
            return JsonResponse({"error": "Anomaly data file not found"}, status=404)
        
        # Read and parse the JSON file
        with open(anomaly_file_path, 'r') as file:
            anomaly_data = json.load(file)
        
        # Limit to first 100 anomalies for performance
        # You can adjust this number or add pagination
        limited_data = anomaly_data[:100] if len(anomaly_data) > 100 else anomaly_data
        
        return JsonResponse(limited_data, safe=False)
        
    except FileNotFoundError:
        return JsonResponse({"error": "Anomaly data file not found"}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format in anomaly data file"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Error loading anomaly data: {str(e)}"}, status=500)
