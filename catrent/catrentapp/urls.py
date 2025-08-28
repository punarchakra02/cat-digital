from django.contrib import admin
from django.urls import path, include
from catrentapp import views

urlpatterns = [
    path('add/', views.add, name="add"),
    path('delete/<int:pk>/', views.delete_machine, name="delete_machine"),
    path('download/<int:pk>/', views.download_qr, name="download_qr"),
    path('rent/<str:machine_id>/', views.rent_machine, name="rent_machine"),
    path('checkout/<str:equipment_id>/', views.qr_scan_info, name="checkout"),
    path('', views.rental_dashboard, name="rental_dashboard"),
]