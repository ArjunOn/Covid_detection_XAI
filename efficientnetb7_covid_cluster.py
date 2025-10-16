
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import ModelCheckpoint

# ============ GPU SETUP FOR HPC ============
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=12000)]
        )
        print(f"{len(gpus)} GPU(s) available. Memory growth enabled.")
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("No GPU detected. Running on CPU.")

# ============ Data Setup ============
path = "/mnt/home/amudalap/Arjun/COVID-19_Radiography_Dataset"
max_images_per_class = 3616
images, labels = [], []
IMG_SIZE = 224

for class_label in os.listdir(path):
    class_path = os.path.join(path, class_label)
    filenames = os.listdir(class_path)[:max_images_per_class]
    for fname in filenames:
        img_path = os.path.join(class_path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_label)

images = np.array(images, dtype='float32') / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

# ============ Dataset Split ============
X_temp, test_x, y_temp, test_y = train_test_split(images, labels, test_size=0.1, random_state=42, stratify=labels)
train_x, val_x, train_y, val_y = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp)

# ============ Data Pipeline ============
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)

# ============ Model Definition ============
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetB7(include_top=False, weights=None, input_tensor=inputs)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

# ============ Callbacks ============
checkpoint = ModelCheckpoint("efficientnetb7_best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

# ============ Training ============
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[checkpoint], verbose=2)

# ============ Evaluation ============
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

model.save("efficientnetb7_covid_model_final.keras")

# ============ Training History Plot ============
def plot_training_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    prec = history.history['precision']
    val_prec = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, prec, label='Train Precision')
    plt.plot(epochs_range, val_prec, label='Val Precision')
    plt.title('Precision')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, recall, label='Train Recall')
    plt.plot(epochs_range, val_recall, label='Val Recall')
    plt.title('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

plot_training_metrics(history)

# ============ Confusion Matrix & Classification Report ============
y_pred_probs = model.predict(test_x)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_y, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

report = classification_report(y_true, y_pred, target_names=le.classes_)
with open("classification_report.txt", "w") as f:
    f.write(report)
print(report)
