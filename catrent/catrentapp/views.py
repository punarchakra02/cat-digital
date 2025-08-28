from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .forms import MachineForm
from django.utils import timezone
from datetime import timedelta
from .models import Machine, Rental, EquipmentUsage, EquipmentHealth


def add(request):
    """Add and list machines"""
    machines = Machine.objects.all()

    if request.method == "POST":
        form = MachineForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("add")
    else:
        form = MachineForm()

    return render(request, "add_machine.html", {"machines": machines, "form": form})


def delete_machine(request, pk):
    """Delete a machine"""
    machine = get_object_or_404(Machine, pk=pk)
    machine.delete()
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
    machine = get_object_or_404(Machine, equipment_id=machine_id)

    if request.method == "POST":
        user_id = request.POST.get("user_id")
        days = int(request.POST.get("days", 1))
        end_date = timezone.now() + timedelta(days=days)

        rental = Rental.objects.create(
            machine=machine,
            user_id=user_id,
            end_date=end_date,
            active=True
        )
        return redirect("rental_dashboard")

    return render(request, "rent_machine.html", {"machine": machine})


def rental_dashboard(request):
    """Dashboard showing all active rentals with live status"""
    rentals = Rental.objects.filter(active=True)

    context = []
    for rental in rentals:
        usage = rental.machine.usages.last()
        health = rental.machine.health.last()

        context.append({
            "rental": rental,
            "usage": usage,
            "health": health,
        })

    return render(request, "rental_dashboard.html", {"data": context})