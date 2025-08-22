from django.shortcuts import render
from django.contrib import messages
from .forms import DiabetesForm, HeartDiseaseForm, AlzheimerForm
from .ml_models import HealthAIPredictor

predictor = HealthAIPredictor()
predictor.load_models()

def index(request):
    return render(request, 'health_ai/index.html')

def diabetes_prediction(request):
    if request.method == 'POST':
        form = DiabetesForm(request.POST)
        if form.is_valid():
            result = predictor.predict_diabetes(form.cleaned_data)
            return render(request, 'health_ai/results.html', {'result': result, 'input_data': form.cleaned_data, 'disease_type': 'diabetes'})
    else:
        form = DiabetesForm()
    return render(request, 'health_ai/diabetes_form.html', {'form': form})

def heart_prediction(request):
    if request.method == 'POST':
        form = HeartDiseaseForm(request.POST)
        if form.is_valid():
            result = predictor.predict_heart_disease(form.cleaned_data)
            return render(request, 'health_ai/results.html', {'result': result, 'input_data': form.cleaned_data, 'disease_type': 'heart'})
    else:
        form = HeartDiseaseForm()
    return render(request, 'health_ai/heart_form.html', {'form': form})

def alzheimer_prediction(request):
    if request.method == 'POST':
        form = AlzheimerForm(request.POST)
        if form.is_valid():
            result = predictor.predict_alzheimer(form.cleaned_data)
            return render(request, 'health_ai/results.html', {'result': result, 'input_data': form.cleaned_data, 'disease_type': 'alzheimer'})
    else:
        form = AlzheimerForm()
    return render(request, 'health_ai/alzheimer_form.html', {'form': form})
