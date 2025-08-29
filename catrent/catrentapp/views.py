import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, FileResponse
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse
from datetime import datetime
from datetime import timedelta
import os
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .utils import generate_operator_id
from .models import Machine, Rental, EquipmentHealth, Operator
from .forms import MachineForm, CheckoutForm, OperatorForm
from .demand_forecasting import equipment_demand_forecast

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
        
        # Count for rented machines by type
        rented_bulldozer_count = 0
        rented_excavator_count = 0
        rented_loader_count = 0
        rented_crane_count = 0
        
        for rental in rented_qs:
            latest_health = rental.machine.health_records.order_by('-timestamp').first()
            rented.append({
                "rental": rental,
                "health": latest_health,
            })
            
            # Count by machine type
            machine_type = rental.machine.type
            if machine_type == 'Bulldozer':
                rented_bulldozer_count += 1
            elif machine_type == 'Excavator':
                rented_excavator_count += 1
            elif machine_type == 'Loader':
                rented_loader_count += 1
            elif machine_type == 'Crane':
                rented_crane_count += 1

        # Available machines
        available = Machine.objects.filter(status='Available')
        
        # Count machines by type
        bulldozer_count = available.filter(type='Bulldozer').count()
        excavator_count = available.filter(type='Excavator').count()
        loader_count = available.filter(type='Loader').count()
        crane_count = available.filter(type='Crane').count()

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
            "bulldozer_count": bulldozer_count,
            "excavator_count": excavator_count,
            "loader_count": loader_count,
            "crane_count": crane_count,
            "rented_bulldozer_count": rented_bulldozer_count,
            "rented_excavator_count": rented_excavator_count,
            "rented_loader_count": rented_loader_count,
            "rented_crane_count": rented_crane_count,
        })

    except Exception as e:
        messages.error(request, 'Error loading dashboard.')
        return render(request, "rental_dashboard.html", {
            "rented": [],
            "available": [],
            "health_chart_data": json.dumps({"labels": [], "data": []}),
            "bulldozer_count": 0,
            "excavator_count": 0,
            "loader_count": 0,
            "crane_count": 0,
            "rented_bulldozer_count": 0,
            "rented_excavator_count": 0,
            "rented_loader_count": 0,
            "rented_crane_count": 0,
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

    # Get all operators and prepare JSON for JS
    operators = Operator.objects.all()
    operators_json = json.dumps([
        {"id": op.operator_id, "name": op.name, "email": op.email}
        for op in operators
    ])

    # If machine is already rented
    if current_rental:
        return render(request, 'checkout.html', {
            'machine': machine,
            'current_rental': current_rental,
            'is_already_rented': True,
            'operators': operators,
            'operators_json': operators_json,
        })

    # POST - create rental
    if request.method == "POST":
        operator_id = request.POST.get("operator_id")
        operator = Operator.objects.get(operator_id=operator_id)

        site_id = request.POST.get("site_id")
        site_name = request.POST.get("site_name", "")

        # Convert strings from datetime-local input to aware datetime objects
        checkout_date_str = request.POST.get("checkout_date")
        expected_return_date_str = request.POST.get("expected_return_date")

        checkout_date = timezone.make_aware(datetime.strptime(checkout_date_str, "%Y-%m-%dT%H:%M"))
        expected_return_date = timezone.make_aware(datetime.strptime(expected_return_date_str, "%Y-%m-%dT%H:%M"))

        # Create Rental
        rental = Rental.objects.create(
            machine=machine,
            operator_id=operator.operator_id,
            operator_name=operator.name,
            site_id=site_id,
            site_name=site_name,
            start_date=checkout_date,
            expected_end_date=expected_return_date,
            active=True,
            status='Active',
        )

        # Update machine status
        machine.status = 'Rented'
        machine.save()

        messages.success(request, f'Equipment {machine.equipment_id} checked out!')
        return redirect('rental_dashboard')

    # GET - show checkout form
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
        'operators': operators,
        'operators_json': operators_json,
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


def generate_forecast(request, equipment_type):
    """Generate demand forecast for a specific equipment type"""
    try:
        # Create forecast directory if it doesn't exist
        forecast_dir = os.path.join(settings.BASE_DIR, 'forecast')
        os.makedirs(forecast_dir, exist_ok=True)
        
        # Get data from the Rental model
        rentals = Rental.objects.all().select_related('machine')
        
        # Convert to DataFrame
        data = []
        for rental in rentals:
            data.append({
                'equipment_id': rental.machine.equipment_id,
                'equipment_type': rental.machine.type,
                'checkout_date': rental.start_date,
                'checkin_date': rental.expected_end_date,
                'site_id': rental.site_id,
                'rate_per_day': float(rental.rate_per_day),
                'maintenance_count': rental.maintenance,
                'engine_hours': rental.engine_hours,
                'idle_hours': rental.idle_hours,
                'year_week': rental.week,
                'target_checkout_count': rental.target_checkout_count,
                'checkout_count_t-1': rental.checkout_count_t_1,
                'checkout_count_t-4': rental.checkout_count_t_4,
                'rolling_mean_4w': rental.rolling_mean_4w,
                'price_index': rental.price_index,
                'pct_maint': rental.pct_maint,
                'engine_hours_avg': rental.engine_hours_avg,
                'idle_ratio_avg': rental.idle_ratio_avg,
                'equipment_type_encoded': rental.equipment_type_encoded,
                'usage_efficiency': rental.usage_efficiency,
                'rental_duration': rental.rental_duration,
                'rate_relative': rental.rate_relative,
                'recent_maintenance': rental.recent_maintenance,
                'maintenance_per_day': rental.maintenance_per_day,
                'engine_hours_per_day': rental.engine_hours_per_day,
                'month_sin': rental.month_sin,
                'month_cos': rental.month_cos,
                'week_sin': rental.week_sin,
                'week_cos': rental.week_cos,
                'rolling_std_4w': rental.rolling_std_4w,
                'rolling_max_4w': rental.rolling_max_4w
            })
        
        df = pd.DataFrame(data)
        
        # Only proceed if we have data
        if len(df) == 0:
            return JsonResponse({
                'status': 'error',
                'message': 'No rental data available for forecasting'
            })
        
        # Generate forecast
        forecast_result = equipment_demand_forecast(df, output_folder=forecast_dir)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Forecast generated for {equipment_type}',
            'image_url': f'/forecast_image/{equipment_type}/',
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({
            'status': 'error',
            'message': f'Error generating forecast: {str(e)}'
        }, status=500)


def get_forecast_image(request, equipment_type):
    """Return the forecast image for a specific equipment type"""
    forecast_dir = os.path.join(settings.BASE_DIR, 'forecast')
    image_path = os.path.join(forecast_dir, f"{equipment_type}_demand_forecast_2025.png")
    
    if os.path.exists(image_path):
        return FileResponse(open(image_path, 'rb'), content_type='image/png')
    else:
        # If individual image doesn't exist, try the combined image
        combined_path = os.path.join(forecast_dir, "equipment_demand_forecast_2025_combined.png")
        if os.path.exists(combined_path):
            return FileResponse(open(combined_path, 'rb'), content_type='image/png')
        
        # Return a 404 if no image is found
        return HttpResponse("Forecast image not found", status=404)

def add_operator(request):
    if request.method == "POST":
        form = OperatorForm(request.POST)
        if form.is_valid():
            operator = form.save(commit=False)
            operator.operator_id = generate_operator_id()
            operator.save()
            messages.success(request, f"Operator {operator.operator_id} added successfully!")
            return redirect("add_operator")
    else:
        form = OperatorForm()
    return render(request, "add_operator.html", {"form": form})

