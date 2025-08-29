from django.contrib import admin
from django.urls import path, include
from catrentapp import views

urlpatterns = [
    path('add/', views.add, name="add"),
    path('delete/<int:pk>/', views.delete_machine, name="delete_machine"),
    path('download/<int:pk>/', views.download_qr, name="download_qr"),
    path('checkout/<str:equipment_id>/', views.qr_scan_info, name="checkout"),
    path('checkin/<int:rental_id>/', views.checkin_machine, name="checkin_machine"),
    path('forecast/<str:equipment_type>/', views.generate_forecast, name="generate_forecast"),
    path('forecast_image/<str:equipment_type>/', views.get_forecast_image, name="get_forecast_image"),
    path('anomaly-dashboard/', views.anomaly_dashboard, name="anomaly_dashboard"),
    path('', views.rental_dashboard, name="rental_dashboard"),
    path("add-operator/", views.add_operator, name="add_operator"),
    
]