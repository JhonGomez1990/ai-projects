# Informe técnico: clasificación de prendas con Fashion-MNIST

## 1. Resumen

Este proyecto compara tres arquitecturas de redes neuronales convolucionales (CNN) para clasificar imágenes de prendas del dataset Fashion-MNIST. Se selecciona el modelo con mejor accuracy de validación y se prueba mediante una interfaz en el notebook que permite elegir una imagen local; también se conserva una opción opcional de cámara.

El mejor resultado correspondió a la CNN sencilla (`CNN_simple`), con 91,17 % de accuracy en validación y 91,22 % en prueba.

## 2. Dataset y preparación

Se utilizaron los archivos CSV locales de Fashion-MNIST, con una etiqueta y 784 valores de píxeles por fila. Cada imagen se transforma a escala de grises y forma `28 × 28 × 1`; los valores de píxel se normalizan dividiéndolos entre 255.

El conjunto de entrenamiento original se separó estratificadamente en 54.000 imágenes para entrenamiento y 6.000 para validación. El conjunto de prueba, separado del entrenamiento, contiene 10.000 imágenes. Las diez clases representan prendas y accesorios: camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, tenis, bolso y botín.

## 3. Metodología y arquitecturas

Las tres redes se implementaron con TensorFlow/Keras y se entrenaron con el optimizador Adam, la función de pérdida `sparse_categorical_crossentropy` y lotes de 128 imágenes. Se definió un máximo de cinco épocas y parada temprana (`EarlyStopping`) usando el accuracy de validación, con restauración de los mejores pesos.

- **CNN sencilla:** dos convoluciones con 32 y 64 filtros, cada una seguida por max pooling; termina con una capa densa de 128 unidades y dropout.
- **CNN profunda:** cuatro convoluciones organizadas en dos bloques; emplea batch normalization, max pooling, dropout y global average pooling.
- **CNN separable:** tres capas `SeparableConv2D`, con max pooling y global average pooling para reducir el costo de las convoluciones.

La selección se realizó por el mayor accuracy de validación. Las métricas de prueba se usaron para comparar el desempeño final y no para escoger el modelo.

## 4. Resultados

| Modelo | Accuracy validación | Accuracy prueba | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|---:|
| CNN sencilla | 91,17 % | 91,22 % | 91,58 % | 91,22 % | 91,31 % |
| CNN profunda | 88,12 % | 88,33 % | 88,97 % | 88,33 % | 88,40 % |
| CNN separable | 75,02 % | 75,11 % | 74,70 % | 75,11 % | 74,34 % |

La CNN sencilla obtuvo el mejor accuracy de validación y de prueba, por lo que se seleccionó para la aplicación. El notebook presenta las curvas de pérdida y accuracy del modelo seleccionado y una matriz de confusión con los aciertos y errores por clase.

## 5. Aplicación desarrollada

El notebook ofrece dos modos de inferencia:

1. **Imagen local, sin webcam:** se abre un selector para elegir una imagen JPG, JPEG o PNG. La imagen se convierte a escala de grises, se ajusta a `28 × 28` y se normaliza antes de la predicción. Se muestran la imagen original, la versión procesada, la clase predicha y la confianza estimada.
2. **Cámara opcional:** OpenCV abre la cámara local. Al presionar espacio se procesa el fotograma y se presenta la clase y su confianza; `q` cierra la ventana.

El modelo seleccionado se guarda como `best_fashion_cnn.keras`, junto con los nombres de las clases en `class_names.json`. La confianza es la probabilidad softmax de la predicción; no debe interpretarse como garantía de acierto.

## 6. Conclusiones y mejoras

En esta ejecución, la arquitectura sencilla superó a las variantes profunda y separable. El resultado sugiere que, para este conjunto y configuración, una arquitectura compacta fue suficiente y las alternativas no mejoraron la generalización.

Como trabajo futuro se propone probar más configuraciones de filtros, dropout y épocas; aplicar aumento de datos; analizar por separado los errores de la matriz de confusión; y probar fotografías reales con fondos e iluminación variados. También convendría mejorar el recorte y centrado de la prenda, así como calibrar la confianza antes de interpretarla como probabilidad fiable.

## 7. Reproducción

Instalar las dependencias listadas en `requirements.txt`, abrir `fashion_mnist_cnn.ipynb` y ejecutar las celdas de carga, entrenamiento, comparación y guardado en orden. Para probar una imagen sin webcam, ejecutar la sección **Probar una imagen local (sin cámara)**. La sección de cámara es opcional y requiere un kernel local con cámara disponible.
