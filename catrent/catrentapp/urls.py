from django.contrib import admin
from django.urls import path, include
from catrentapp import views

urlpatterns = [
    path('add/', views.add, name="add"),
    path('delete/<int:pk>/', views.delete_machine, name="delete_machine"),
    path('download/<int:pk>/', views.download_qr, name="download_qr"),
    path('checkout/<str:equipment_id>/', views.qr_scan_info, name="checkout"),
    path('checkin/<int:rental_id>/', views.checkin_machine, name="checkin_machine"),
    path('anomaly-dashboard/', views.anomaly_dashboard, name="anomaly_dashboard"),
    path('anomaly-data/', views.anomaly_data, name="anomaly_data"),
    path('', views.rental_dashboard, name="rental_dashboard"),
]