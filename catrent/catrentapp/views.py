from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from .forms import MachineForm, RentalForm, CheckoutForm
from django.utils import timezone
from datetime import timedelta
from .models import Machine, Rental, EquipmentUsage, EquipmentHealth
from django.contrib import messages


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
                messages.error(request, f'Error adding machine: {str(e)}')
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
        messages.error(request, f'Error deleting machine: {str(e)}')
    return redirect("add")


def download_qr(request, pk):
    """Download the QR code image for a machine"""
    machine = get_object_or_404(Machine, pk=pk)
    if not machine.qr_code:
        return HttpResponse("No QR code available for this equipment.")

    file_path = machine.qr_code.path
    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="image/png")
        response["Content-Disposition"] = f'attachment; filename="qr_{machine.equipment_id}.png"'
        return response



def rent_machine(request, machine_id):
    """Admin scans QR and rents a machine"""
    try:
        machine = get_object_or_404(Machine, equipment_id=machine_id)

        # Check if machine is already rented
        if machine.status == 'Rented':
            messages.error(request, 'This machine is already rented!')
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
                rental_cost_per_day=machine.list_price_per_day,
                active=True,
                status='Active'
            )
            messages.success(request, f'Machine {machine.equipment_id} rented successfully!')
            return redirect("rental_dashboard")

        return render(request, "rent_machine.html", {"machine": machine})
        
    except Exception as e:
        messages.error(request, f'Error renting machine: {str(e)}')
        return redirect("rental_dashboard")


def rental_dashboard(request):
    """Dashboard showing all active rentals with live status"""
    try:
        rentals = Rental.objects.filter(status='Active').select_related('machine')

        context = []
        for rental in rentals:
            # Get latest usage data
            latest_usage = rental.machine.usages.order_by('-date').first()
            # Get latest health record
            latest_health = rental.machine.health_records.order_by('-timestamp').first()
            
            context.append({
                "rental": rental,
                "usage": latest_usage,
                "health": latest_health,
            })

        return render(request, "rental_dashboard.html", {"data": context})
        
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {str(e)}')
        return render(request, "rental_dashboard.html", {"data": []})


def qr_scan_info(request, equipment_id):
    """QR scan checkout view for admins"""
    try:
        machine = get_object_or_404(Machine, equipment_id=equipment_id)
        
        # Check if equipment is already rented
        current_rental = machine.current_rental
        if current_rental:
            # Equipment is already rented - show rental info instead of checkout form
            context = {
                'machine': machine,
                'current_rental': current_rental,
                'is_already_rented': True,
            }
            return render(request, 'checkout.html', context)
        
        # Equipment is available for checkout
        if request.method == "POST":
            form = CheckoutForm(request.POST)
            if form.is_valid():
                # Double check that machine is still available
                if machine.current_rental:
                    messages.error(request, f'Equipment {machine.equipment_id} was just rented by someone else!')
                    return redirect('checkout', equipment_id=equipment_id)
                
                # Create rental record
                rental = Rental.objects.create(
                    machine=machine,
                    operator_id=form.cleaned_data['operator_id'],
                    operator_name=form.cleaned_data['operator_name'],
                    site_id=form.cleaned_data['site_id'],
                    site_name=form.cleaned_data['site_name'],
                    start_date=form.cleaned_data['checkout_date'],
                    expected_end_date=form.cleaned_data['expected_return_date'],
                    active=True
                )
                
                # Update machine status
                machine.status = 'Rented'
                machine.save()
                
                messages.success(request, f'Equipment {machine.equipment_id} successfully checked out!')
                return redirect('rental_dashboard')
        else:
            # Pre-fill form with machine data and current time
            initial_data = {
                'equipment_id': machine.equipment_id,
                'type': machine.type,
                'status': machine.status,
                'checkout_date': timezone.now().strftime('%Y-%m-%dT%H:%M'),
            }
            form = CheckoutForm(initial=initial_data)
        
        context = {
            'machine': machine,
            'form': form,
            'is_already_rented': False,
        }
        
        return render(request, 'checkout.html', context)
        
    except Exception as e:
        messages.error(request, f'Error processing checkout: {str(e)}')
        return redirect('add')