from django import forms
from .models import Machine, Rental, EquipmentUsage, Operator

class MachineForm(forms.ModelForm):
    class Meta:
        model = Machine
        fields = ['equipment_id', 'type', 'rate_per_day']
        widgets = {
            'equipment_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., EQX1001'}),
            'type': forms.Select(attrs={'class': 'form-control'}),
        }


class RentalForm(forms.ModelForm):
    class Meta:
        model = Rental
        fields = ['operator_id', 'operator_name', 'site_id', 'site_name', 'expected_end_date']
        widgets = {
            'operator_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Operator ID'}),
            'operator_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Operator Name'}),
            'site_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Site ID'}),
            'site_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Site Name'}),
            'expected_end_date': forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
        }


class UsageLogForm(forms.ModelForm):
    class Meta:
        model = EquipmentUsage
        fields = ['engine_hours', 'idle_hours', 'fuel_consumed', 'distance_traveled', 'work_type', 'notes']
        widgets = {
            'engine_hours': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'idle_hours': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'fuel_consumed': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'distance_traveled': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'work_type': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Type of work performed'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }


class CheckoutForm(forms.Form):
    equipment_id = forms.CharField(
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'readonly': 'readonly'})
    )
    type = forms.CharField(
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'readonly': 'readonly'})
    )
    status = forms.CharField(
        max_length=50, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'readonly': 'readonly'})
    )
    operator_id = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Operator ID', 'required': True})
    )
    operator_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Operator Name', 'required': True})
    )
    site_id = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Site ID', 'required': True})
    )
    site_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Site Name (optional)'})
    )
    checkout_date = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local', 'required': True})
    )
    expected_return_date = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local', 'required': True})
    )


class OperatorForm(forms.ModelForm):
    class Meta:
        model = Operator
        fields = ['name', 'email']
