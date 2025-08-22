import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import json

DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'data/models'

X_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_train_alzheimer.npy'))
y_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_train_alzheimer.npy'))
X_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_test_alzheimer.npy'))
y_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_test_alzheimer.npy'))

num_classes = len(set(y_train))

# Convert to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)

metrics = {
    'accuracy': accuracy
}

print("Alzheimer Model Metrics:", metrics)

model.save(os.path.join(MODELS_DIR, 'alzheimer_model.h5'))

with open(os.path.join(MODELS_DIR, 'alzheimer_metrics.json'), 'w') as f:
    json.dump(metrics, f)
