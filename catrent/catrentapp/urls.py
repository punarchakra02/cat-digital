from django.contrib import admin
from django.urls import path, include
from catrentapp import views

urlpatterns = [
    path('', views.add, name="add"),
    path('delete/<int:pk>/', views.delete_machine, name="delete_machine"),
    path('download/<int:pk>/', views.download_qr, name="download_qr"),
]