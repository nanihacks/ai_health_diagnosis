# AI Health Diagnosis Web Application

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Django](https://img.shields.io/badge/Django-5.2.5-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## Project Overview

This is a web application built with **Django** that provides AI-powered health risk assessments for **Diabetes**, **Heart Disease**, and **Alzheimer’s**. The app uses machine learning models trained on popular medical datasets to deliver quick, accurate risk predictions with actionable recommendations.

---

## Features

- User-friendly, responsive UI with clean design and smooth animations
- Risk prediction for Diabetes, Heart Disease, and Alzheimer’s
- Interactive forms with client-side validation
- Detailed result pages showing risk level, probability, and health advice
- Modular Django application structure with trained ML models integration
- User Authentication (optional - can be added)
- Prediction history logging (optional - can be added)
- Easily extensible for additional diseases or enhanced ML models

---

## Technologies Used

- **Python 3.13**  
- **Django 5.2.5**  
- **TensorFlow 2.x** for ML model inference  
- **Joblib** for loading scalers  
- HTML, CSS, JavaScript (custom, responsive design)  
- Bootstrap (optional) or custom CSS animations

---

## Setup and Installation

1. Clone the repository:


2. Create and activate a virtual environment:


3. Install required packages:


4. Run migrations:


5. Collect static files:


6. Run the development server:


7. Open your browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and start using the app!

---

## Project Structure

- `health_ai/` — Main Django app containing views, forms, models, and ML integration
- `ml_training/` — Scripts for training the ML models (Diabetes, Heart, Alzheimer’s)
- `data/` — Dataset files, processed data, saved ML models and scalers
- `templates/health_ai/` — HTML templates
- `static/css/`, `static/js/` — Custom styling and JavaScript
- `ai_health_diagnosis/` — Project settings and configurations

---

## How It Works

- User fills in specific health-related data on the form page.
- The input is validated and preprocessed using saved scalers.
- The respective TensorFlow ML model predicts disease risk probabilities.
- The app displays risk levels, probability, and recommendations interactively.
- Optionally, users can log in and save prediction history (feature extensible).

---

## Future Enhancements

- Full user authentication and user dashboards  
- Logging and analytics of prediction usage  
- Integration with health APIs for real-time data  
- Model explainability with SHAP or LIME  
- Deployment scripts for cloud hosting (Heroku, DigitalOcean, AWS)

---

## License

This project is open source under the MIT License.

---

## Acknowledgments

- Thanks to dataset providers such as Kaggle and UCI ML Repository  
- Inspired by multiple open-source Django & ML projects  

---

## Contact

For questions or support, contact:  
**Praveen** – annampraveen2003.com  
GitHub: https://github.com/nanihacks/ai_health_diagnosis
