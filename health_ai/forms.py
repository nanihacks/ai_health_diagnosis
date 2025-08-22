from django import forms


class DiabetesForm(forms.Form):
    pregnancies = forms.IntegerField(min_value=0)
    glucose = forms.FloatField(min_value=0)
    blood_pressure = forms.FloatField(min_value=0)
    skin_thickness = forms.FloatField(min_value=0)
    insulin = forms.FloatField(min_value=0)
    bmi = forms.FloatField(min_value=0)
    diabetes_pedigree_function = forms.FloatField(min_value=0)
    age = forms.IntegerField(min_value=0)


class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(min_value=0)
    sex = forms.ChoiceField(choices=[(1, 'Male'), (0, 'Female')])
    chest_pain_type = forms.ChoiceField(choices=[(0, 'Typical Angina'), (1, 'Atypical Angina'), (2, 'Non-Anginal Pain'), (3, 'Asymptomatic')])
    resting_bp = forms.FloatField(min_value=0)
    cholesterol = forms.FloatField(min_value=0)
    fasting_bs = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    resting_ecg = forms.ChoiceField(choices=[(0, 'Normal'), (1, 'Abnormal'), (2, 'Hypertrophy')])
    max_hr = forms.FloatField(min_value=0)
    exercise_angina = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    oldpeak = forms.FloatField(min_value=0)
    st_slope = forms.ChoiceField(choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')])



class AlzheimerForm(forms.Form):
    age = forms.IntegerField(min_value=0)
    gender = forms.ChoiceField(choices=[(0, 'Female'), (1, 'Male')])
    education_level = forms.ChoiceField(choices=[(0, 'None'), (1, 'High School'), (2, 'Bachelor'), (3, 'Higher')])
    bmi = forms.FloatField(min_value=0)
    smoking = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    alcohol_consumption = forms.FloatField(min_value=0)
    physical_activity = forms.FloatField(min_value=0)
    diet_quality = forms.FloatField(min_value=0)
    sleep_quality = forms.FloatField(min_value=0)
    family_history_alzheimers = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    cardiovascular_disease = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    depression = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    head_injury = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    hypertension = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    diabetes = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    stroke = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    systolic_bp = forms.FloatField(label="Systolic Blood Pressure", min_value=0)
    diastolic_bp = forms.FloatField(label="Diastolic Blood Pressure", min_value=0)

