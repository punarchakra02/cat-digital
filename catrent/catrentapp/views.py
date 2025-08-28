from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Machine
from .forms import MachineForm

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
