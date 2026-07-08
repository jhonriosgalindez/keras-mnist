# Documentación del Proyecto: Clasificación de Dígitos con Keras y MNIST

## Descripción
Este proyecto implementa una red neuronal artificial secuencial para el reconocimiento y clasificación de dígitos escritos a mano. Utiliza el popular conjunto de datos MNIST y la API de Keras integrada en TensorFlow para construir, entrenar y evaluar un modelo de aprendizaje profundo.

## Tecnologías y Librerías Utilizadas
* **Python 3**: Lenguaje base para la ejecución del código.
* **TensorFlow y Keras**: Empleados para la definición del modelo secuencial (`Sequential`), la adición de capas (`Dense`, `Flatten`), la codificación de etiquetas (`to_categorical`) y la importación directa del dataset (`mnist`).
* **Matplotlib (`pyplot`)**: Utilizado para la visualización gráfica de las imágenes del conjunto de datos y la exposición de los resultados de las predicciones.
* **NumPy**: Usado para la manipulación de arreglos numéricos, expansión de dimensiones (`expand_dims`) y la obtención de la clase predicha mediante la función `argmax`.

## Fuente de Datos
Los datos se importan directamente a través de la librería de TensorFlow (`tensorflow.keras.datasets.mnist.load_data()`). El conjunto de datos de entrenamiento analizado consta de 60,000 imágenes en escala de grises, cada una con una resolución de 28x28 píxeles. Las etiquetas correspondientes se cargan en un vector de 60,000 elementos que indican el valor numérico (0-9) esperado para cada imagen.

## Estructura del Análisis

### 1. Exploración Inicial y Visualización
* Se realiza la carga exclusiva de los datos de entrenamiento (`X_train`, `y_train`).
* Mediante un ciclo iterativo y el uso de `matplotlib`, se genera una figura que muestra visualmente las 10 primeras imágenes del dataset original, desplegando como título de cada una su etiqueta real.

### 2. Preprocesamiento de Datos
* **Normalización:** Los valores numéricos de los píxeles de las imágenes (que naturalmente van de 0 a 255) se dividen entre 255 para convertirlos a un rango de 0 a 1, optimizando el rendimiento computacional de la red neuronal.
* **Codificación One-Hot:** Las etiquetas numéricas originales se transforman en matrices binarias distribuidas en 10 clases usando la función `to_categorical`.

### 3. Construcción del Modelo Secuencial
Se diseña una arquitectura de red neuronal encadenada que consta de las siguientes capas:
* **Capa de Entrada (Flatten):** Toma la matriz bidimensional de cada imagen (28x28) y la aplana, convirtiéndola en un único vector unidimensional.
* **Capa Oculta (Dense):** Capa principal densamente conectada compuesta por 128 neuronas y equipada con una función de activación `relu`.
* **Capa de Salida (Dense):** Capa final configurada con 10 neuronas (una para cada dígito posible) empleando una función de activación `softmax` para dictaminar probabilidades categóricas.

### 4. Compilación y Entrenamiento
* El modelo se compila utilizando el optimizador `adam` (aunque el código detalla la posibilidad de usar descenso de gradiente) y la función de pérdida `categorical_crossentropy`. Además, se define el seguimiento de la métrica de exactitud (`accuracy`).
* El entrenamiento se programa para 10 iteraciones (`epochs`). Los resultados reportan un aprendizaje altamente eficiente, partiendo de una precisión inicial en la primera iteración (87.60%) hasta alcanzar una exactitud del 99.62% sobre los datos de entrenamiento al concluir la iteración final, reduciendo la función de pérdida a 0.0123.

### 5. Predicción y Evaluación Individual
* Para comprobar la funcionalidad, se aísla una imagen de prueba (índice 7 del dataset), graficándola en pantalla con su valor esperado (el número 3).
* Se añade una dimensión adicional al arreglo usando `np.expand_dims` para simular un lote o *batch* compatible con el modelo, transformando sus proporciones de `(28, 28)` a `(1, 28, 28)`.
* Se invoca el método `.predict()` y el modelo identifica correctamente la imagen de prueba, imprimiendo en consola que la etiqueta predicha es el 3.
