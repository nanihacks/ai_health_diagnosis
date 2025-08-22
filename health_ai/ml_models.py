import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from django.conf import settings


class HealthAIPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}

        self.feature_names = {
            'diabetes': ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                         'insulin', 'bmi', 'diabetes_pedigree_function', 'age'],

            'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],

            'alzheimer': ['Age', 'Gender', 'EducationLevel', 'BMI', 'Smoking',
                          'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
                          'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                          'Depression', 'HeadInjury', 'Hypertension', 'Diabetes', 'Stroke',
                          'ADL', 'BehavioralProblems', 'CholesterolTotal',
                          'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                          'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'Confusion',
                          'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                          'Forgetfulness', 'SystolicBP', 'DiastolicBP']
        }

    def load_models(self):
        models_path = settings.ML_MODELS_PATH

        for model_name in ['diabetes', 'heart', 'alzheimer']:
            model_path = os.path.join(models_path, f"{model_name}_model.h5")
            if os.path.exists(model_path):
                self.models[model_name] = tf.keras.models.load_model(model_path)

        scalers_path = os.path.join(models_path, 'scalers')
        for model_name in ['diabetes', 'heart', 'alzheimer']:
            scaler_file = os.path.join(scalers_path, f"{model_name}_scaler.joblib")
            if os.path.exists(scaler_file):
                self.scalers[model_name] = joblib.load(scaler_file)

    def preprocess_input(self, input_data, model_name):
        if model_name == 'heart':
            feature_map = {
                'age': 'age',
                'sex': 'sex',
                'chest_pain_type': 'cp',
                'resting_bp': 'trestbps',
                'cholesterol': 'chol',
                'fasting_bs': 'fbs',
                'resting_ecg': 'restecg',
                'max_hr': 'thalach',
                'exercise_angina': 'exang',
                'oldpeak': 'oldpeak',
                'st_slope': 'slope',
                'ca': 'ca',
                'thal': 'thal',
            }
            mapped_data = {}
            for user_key, train_key in feature_map.items():
                if user_key in input_data:
                    mapped_data[train_key] = input_data[user_key]
                else:
                    if train_key in ['ca', 'thal']:
                        mapped_data[train_key] = 0
                    else:
                        raise ValueError(f"Missing required feature '{user_key}' for heart prediction")

            df = pd.DataFrame([mapped_data])
            scaler = self.scalers.get(model_name)
            if scaler:
                df = df[scaler.feature_names_in_]
                return scaler.transform(df)
            return df.values

        elif model_name == 'alzheimer':
            expected_features = self.feature_names['alzheimer']

            form_to_scaler_map = {
                'age': 'Age',
                'gender': 'Gender',
                'education_level': 'EducationLevel',
                'bmi': 'BMI',
                'smoking': 'Smoking',
                'alcohol_consumption': 'AlcoholConsumption',
                'physical_activity': 'PhysicalActivity',
                'diet_quality': 'DietQuality',
                'sleep_quality': 'SleepQuality',
                'family_history_alzheimers': 'FamilyHistoryAlzheimers',
                'cardiovascular_disease': 'CardiovascularDisease',
                'depression': 'Depression',
                'head_injury': 'HeadInjury',
                'hypertension': 'Hypertension',
                'diabetes': 'Diabetes',
                'stroke': 'Stroke',
                'systolic_bp': 'SystolicBP',
                'diastolic_bp': 'DiastolicBP',
                # Add other features mapping as needed
            }

            mapped_data = {}
            for feat in expected_features:
                form_key = next((k for k, v in form_to_scaler_map.items() if v == feat), None)
                if form_key and form_key in input_data:
                    mapped_data[feat] = input_data[form_key]
                else:
                    mapped_data[feat] = 0

            if 'Gender' in mapped_data:
                if isinstance(mapped_data['Gender'], str):
                    mapped_data['Gender'] = 1 if mapped_data['Gender'].lower() == 'male' else 0

            df = pd.DataFrame([mapped_data])
            scaler = self.scalers.get(model_name)
            if scaler:
                df = df[scaler.feature_names_in_]
                return scaler.transform(df)
            return df.values

        else:
            expected_features = self.feature_names.get(model_name)
            df = pd.DataFrame([input_data])
            if expected_features:
                df = df.reindex(columns=expected_features)
            scaler = self.scalers.get(model_name)
            if scaler:
                return scaler.transform(df)
            return df.values

    def predict_diabetes(self, input_data):
        X = self.preprocess_input(input_data, 'diabetes')
        model = self.models['diabetes']
        proba = model.predict(X)[0]
        prediction = int(proba > 0.5)
        return {
            'prediction': prediction,
            'probability': float(proba),
            'risk_level': self._get_risk_level(proba),
            'recommendation': self._get_recommendation('diabetes', prediction),
        }

    def predict_heart_disease(self, input_data):
        X = self.preprocess_input(input_data, 'heart')
        model = self.models['heart']
        proba = model.predict(X)[0]
        prediction = int(proba > 0.5)
        return {
            'prediction': prediction,
            'probability': float(proba),
            'risk_level': self._get_risk_level(proba),
            'recommendation': self._get_recommendation('heart', prediction),
        }

    def predict_alzheimer(self, input_data):
        X = self.preprocess_input(input_data, 'alzheimer')
        model = self.models['alzheimer']
        preds = model.predict(X)[0]
        pred_class = int(np.argmax(preds))
        proba = float(preds[pred_class])
        return {
            'prediction': pred_class,
            'probabilities': preds.tolist(),
            'probability': proba,
            'risk_level': self._get_risk_level(proba),
            'recommendation': self._get_alzheimer_recommendation(pred_class),
        }

    def _get_risk_level(self, proba):
        if proba < 0.3:
            return 'Low'
        elif proba < 0.7:
            return 'Medium'
        else:
            return 'High'

    def _get_recommendation(self, disease, pred):
        if disease == 'diabetes':
            return (
                'Low risk for diabetes. Maintain healthy lifestyle.' if pred == 0 
                else 'High risk for diabetes. Consult your doctor.'
            )
        if disease == 'heart':
            return (
                'Low risk for heart disease. Maintain healthy lifestyle.' if pred == 0 
                else 'High risk for heart disease. Consult your doctor.'
            )
        return 'Consult your healthcare provider for detailed advice.'

    def _get_alzheimer_recommendation(self, pred_class):
        recs = [
            "No dementia detected. Maintain brain health.",
            "Mild dementia detected. Monitor cognitive health.",
            "Moderate dementia detected. Consult a neurologist.",
            "Severe dementia detected. Seek immediate medical care."
        ]
        return recs[pred_class] if pred_class < len(recs) else "Consult a healthcare professional."
