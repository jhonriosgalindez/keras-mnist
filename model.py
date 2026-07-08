# ==========================================
# PROYECTO: CLASIFICACIÓN DE DÍGITOS MNIST
# ==========================================

# 1. Importar todas las dependencias necesarias
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

print("Cargando datos de MNIST...")

# 2. Cargar los datos
(x_train, y_train), _ = mnist.load_data()

print(f"Forma de las imágenes de entrenamiento (X_train): {x_train.shape}")
print(f"Forma de las etiquetas de entrenamiento (y_train): {y_train.shape}")

# 3. Visualizar las 10 primeras imágenes del dataset original
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray') # Dibuja la imagen en escala de grises
    plt.axis('off')
    plt.title(str(y_train[i])) # Coloca el número real como título
plt.suptitle("Muestra de las 10 primeras imágenes del Dataset")
plt.show()

# 4. Preprocesamiento de los datos
# Normalizar los datos: Los valores de los píxeles van de 0 a 255. 
# Al dividir entre 255, los valores quedan en un rango de 0 a 1.
x_train_norm = x_train / 255.0 

# Codificar etiquetas numéricas en formato one-hot (matrices binarias de 10 clases)
y_train_encoded = to_categorical(y_train, num_classes=10)

# 5. Construir el modelo secuencial de la Red Neuronal
model = Sequential([
    # Capa de entrada: Aplana la matriz de 28x28 píxeles a un vector de 784 elementos
    Flatten(input_shape=(28, 28)),
    
    # Capa oculta: 128 neuronas con función de activación ReLU
    Dense(128, activation='relu'),
    
    # Capa de salida: 10 neuronas (una por dígito) con activación Softmax para probabilidades
    Dense(10, activation='softmax')
])

# 6. Compilar el modelo
model.compile(
    optimizer='adam', # Optimizador Adam 
    loss='categorical_crossentropy', # Función de pérdida para clasificación multiclase
    metrics=['accuracy'] # Métrica a evaluar: exactitud
)

# 7. Entrenar el modelo
print("\nIniciando el entrenamiento del modelo...")
history = model.fit(
    x_train_norm,
    y_train_encoded,
    epochs=10 # Número de iteraciones completas sobre el dataset
)

# 8. Evaluación individual y Predicción
# Escoger una imagen específica del conjunto de entrenamiento
indice = 7
imagen_prueba = x_train_norm[indice] 
etiqueta_esperada = np.argmax(y_train_encoded[indice]) # Obtener el valor real

# Mostrar la imagen seleccionada para prueba
plt.figure(figsize=(3, 3))
plt.imshow(imagen_prueba, cmap='gray')
plt.title(f"Etiqueta real: {etiqueta_esperada}")
plt.axis('off')
plt.show()

# Preparar la imagen para la predicción
# Añadir una dimensión extra para simular un batch de 1 imagen: (28, 28) --> (1, 28, 28)
imagen_input = np.expand_dims(imagen_prueba, axis=0)

# Realizar la predicción con el modelo entrenado
prediccion = model.predict(imagen_input)
clase_predicha = np.argmax(prediccion)

print(f"\n==========================================")
print(f"Etiqueta esperada: {etiqueta_esperada}")
print(f"Etiqueta predicha por la IA: {clase_predicha}")
print(f"==========================================")
