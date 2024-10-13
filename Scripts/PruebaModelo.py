#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow numpy pillow


# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

# Cargar el modelo
modelo = load_model('modelo_comida.h5')

# Directorio de prueba
test_dir = 'imagenes/prueba'

# Preprocesamiento de los datos de prueba
def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0].astype(np.uint8))
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=equalize_histogram
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

# Evaluar el modelo en los datos de prueba
overall_loss, overall_accuracy = modelo.evaluate(test_data, steps=len(test_data))
print(f'Accuracy del modelo: {overall_accuracy:.2f}')

# Predicciones en todos los ejemplos
def mostrar_predicciones(modelo, test_data):
    class_labels = list(test_data.class_indices.keys())
    predictions = []
    images = []
    true_labels = []

    for i in range(len(test_data)):
        x, y_true = test_data[i]
        pred = modelo.predict(x)
        predictions.extend(pred)
        images.extend(x)
        true_labels.extend(y_true)

    plt.figure(figsize=(20, 100))
    cantidad = len(images)  # Mostrar todas las imágenes
    for i in range(cantidad):
        image = np.clip(images[i], 0, 1)  # Ajustar los valores de los píxeles para estar en el rango [0, 1]
        plt.subplot((cantidad // 5) + 1, 5, i + 1)
        plt.imshow(image)
        plt.title(f"Pred: {class_labels[np.argmax(predictions[i])]}\nReal: {class_labels[np.argmax(true_labels[i])]} ")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

mostrar_predicciones(modelo, test_data)


# In[ ]:


Pe

