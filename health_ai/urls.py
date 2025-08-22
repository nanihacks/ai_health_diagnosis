from django.urls import path
from . import views

app_name = "health_ai"

urlpatterns = [
    path('', views.index, name='index'),
    path('diabetes/', views.diabetes_prediction, name='diabetes_prediction'),
    path('heart/', views.heart_prediction, name='heart_prediction'),
    path('alzheimer/', views.alzheimer_prediction, name='alzheimer_prediction'),
]
