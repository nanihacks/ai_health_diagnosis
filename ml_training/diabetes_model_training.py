import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'data/models'

X_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_train_diabetes.npy'))
y_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_train_diabetes.npy'))
X_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_test_diabetes.npy'))
y_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_test_diabetes.npy'))

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

print("Diabetes Model Metrics:", metrics)

model.save(os.path.join(MODELS_DIR, 'diabetes_model.h5'))

with open(os.path.join(MODELS_DIR, 'diabetes_metrics.json'), 'w') as f:
    json.dump(metrics, f)
