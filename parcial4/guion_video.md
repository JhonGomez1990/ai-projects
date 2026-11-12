# Guion del video — Parcial 4

**Duración objetivo:** 9–10 minutos · **Integrantes:** Jhon, Laura y Miguel

**Preparación antes de grabar**
- Tener abiertos: el notebook `parcial4_bank_loan.ipynb` (ya ejecutado), la carpeta `results/figures` y la aplicación corriendo con `streamlit run app.py`.
- Compartir pantalla; cada persona habla en su bloque. Las frases son una guía, pueden decirlas con sus palabras.

---

## 1. Introducción y problema — Miguel (≈1 min)

**En pantalla:** `parcial4/README.md` abierto (o la página del repositorio en GitHub cuando esté subido).

> "Hola, somos Jhon, Laura y Miguel. En este cuarto parcial construimos un sistema que clasifica clientes bancarios usando Deep Learning, pero con una particularidad: los datos del cliente son una tabla, y nosotros los convertimos en una imagen para que una red convolucional los clasifique.
> Usamos el dataset Bank Loan. La pregunta es: ¿este cliente va a pagar su préstamo, *Fully Paid*, o lo va a incumplir, *Charged Off*?
> Les vamos a mostrar los datos, las dos representaciones visuales, los seis modelos que entrenamos, la comparación y la aplicación final."

## 2. Datos y preparación — Miguel (≈1,5 min)

**En pantalla:** notebook `parcial4_bank_loan.ipynb`, sección 1 (tabla de datos, gráficas de clases, plazo y puntaje de crédito) y sección 2 (lista de las 27 variables). Alternativa: `informe_tecnico.md` sección 2.

> "El archivo de entrenamiento trae 100.514 filas. Encontramos 514 filas vacías y unos 18.000 préstamos duplicados por Loan ID; después de limpiar quedan 81.999 clientes. El 72 % paga y el 28 % incumple, así que las clases están desbalanceadas.
> También encontramos errores en los datos: montos de préstamo de 99.999.999, que es un valor de relleno, y puntajes de crédito multiplicados por diez, como 7.120 en vez de 712. Los corregimos.
> Con las 16 columnas originales creamos 27 variables: por ejemplo la relación deuda/ingreso, la utilización del crédito y el préstamo sobre el ingreso. Luego las llevamos todas a una escala de 0 a 1 con una transformación por cuantiles. Dividimos los datos en 70 % entrenamiento, 15 % validación y 15 % prueba."

## 3. Representaciones visuales — Laura (≈1,5 min)

**En pantalla:** notebook, sección 3 (grilla de 2×6 imágenes).

> "Ahora, ¿cómo convertimos un cliente en una imagen? Probamos dos estrategias.
> La primera es la **matriz de características**: ordenamos las 27 variables según su correlación, para que las variables parecidas queden juntas, y las ponemos en una cuadrícula de 6×6. Cada celda se colorea según su valor, de morado (bajo) a amarillo (alto).
> La segunda es el **Gramian Angular Field**: cada valor se convierte en un ángulo y se construye una imagen de 27×27 donde cada píxel representa la relación entre dos variables. Tiene tres canales: la suma de ángulos, la diferencia de ángulos y el producto de las variables. Así la red ve las interacciones entre pares de variables.
> Ambas imágenes se escalan a 32×32 píxeles y se generan directamente en la GPU durante el entrenamiento."

## 4. Arquitecturas y entrenamiento — Laura (≈1,5 min)

**En pantalla:** notebook, tabla de la sección 4 y `src/models.py`.

> "Para cada representación entrenamos tres esquemas, en total seis modelos:
> 1. **Desde cero:** una CNN propia con cuatro bloques de convolución, batch normalization y max pooling, y al final global average pooling.
> 2. **Transfer learning:** una ResNet-18 preentrenada en ImageNet; congelamos toda la red y solo entrenamos la última capa.
> 3. **Fine tuning:** partimos del modelo de transfer learning y descongelamos los últimos bloques, layer3 y layer4, con una tasa de aprendizaje más baja.
> Por el desbalance usamos una función de pérdida ponderada por clase, y parada temprana según el ROC-AUC de validación. Como referencia también entrenamos una regresión logística y un modelo de gradient boosting sobre los datos tabulares."

## 5. Resultados y comparación — Jhon (≈2 min)

**En pantalla:** `results/figures/comparacion_metricas.png`, `roc_prueba.png`, `curvas_entrenamiento.png` y la tabla del notebook.

> "Comparamos los modelos con accuracy, accuracy balanceada, precisión, recall, F1 y ROC-AUC. Le damos más peso al ROC-AUC y al F1 macro porque el accuracy engaña con clases desbalanceadas: un modelo que diga siempre 'paga' ya tendría 72 %.
> Los resultados en prueba: con la matriz de características, desde cero 0,732 de ROC-AUC, fine tuning 0,683 y transfer learning 0,646. Con el GAF, desde cero 0,724, fine tuning 0,667 y transfer learning 0,640. En las dos representaciones el orden es el mismo: gana desde cero, luego fine tuning y al final transfer learning. ¿Por qué? Porque nuestras imágenes son cuadrículas de colores y no se parecen a las fotos de ImageNet, entonces lo que aprendió la ResNet no se transfiere bien. Al descongelar las últimas capas mejora, pero una CNN pequeña entrenada para estas imágenes funciona mejor, y con 43 veces menos parámetros.
> El mejor modelo fue **la matriz de características con la CNN desde cero**, con ROC-AUC de **0,732** y F1 macro de **0,591**, con accuracy balanceada de 66,6 %, en prueba, y por eso es el que usa la aplicación.
> En las curvas de entrenamiento se ve que la pérdida de validación baja junto con la de entrenamiento, sin separarse, así que no hay sobreajuste fuerte, y el ROC-AUC se estabiliza entre la época 5 y la 7.
> Algo importante y honesto: el modelo tabular de referencia, gradient boosting, obtiene prácticamente lo mismo: 0,7316 frente a nuestro 0,7325. Convertir la tabla en imagen funciona, pero no supera automáticamente a un modelo tabular; su ventaja es poder usar redes preentrenadas y tener una representación visual de cada cliente."

**En pantalla:** matriz de confusión.

> "En la matriz de confusión vemos que el modelo detecta el 81 % de los clientes que incumplen, 8 de cada 10, a costa de algunos falsos positivos: clientes buenos que marca como riesgosos."

## 6. Demostración de la aplicación — Jhon (≈1,5 min)

**En pantalla:** aplicación Streamlit.

> "Esta es la aplicación. A la izquierda ingresamos los datos de un cliente nuevo: monto, plazo, puntaje de crédito, ingreso, deudas, etc."

**Acciones:**
1. Mostrar el cliente por defecto: señalar la imagen, la predicción y el nivel de confianza.
2. Presionar **"Cargar cliente aleatorio de credit_test.csv"** dos o tres veces (clientes que el modelo nunca vio).
3. Cambiar a mano: Term = *Long Term*, Credit Score = 600, Monthly Debt alto → mostrar cómo cambia la imagen y sube la probabilidad de *Charged Off*.
4. Abrir el desplegable con la otra representación y la tabla de variables transformadas.

> "La aplicación muestra la representación visual del cliente, que es exactamente lo que ve la red, la predicción y su nivel de confianza, que es la probabilidad que asigna el modelo. Ojo: la confianza no es una garantía de acierto."

## 7. Mejoras y cierre — Miguel (≈1 min)

**En pantalla:** `informe_tecnico.md`, sección 7 (Posibles mejoras).

> "Como mejoras proponemos: probar otras representaciones como DeepInsight o IGTD, que ubican las variables en el plano de forma aprendida; ajustar el umbral de decisión según el costo de negocio de un falso negativo; calibrar la probabilidad para que la confianza sea más confiable; usar validación cruzada y búsqueda de hiperparámetros; y combinar la CNN con el modelo tabular en un ensamble.
> El código, el notebook, el informe y este video están en el repositorio y en la carpeta del Drive. Muchas gracias."

---

### Checklist final
- [ ] Video entre 8 y 10 minutos, con las tres voces.
- [ ] Subir a la carpeta del Drive (subcarpeta con los nombres): notebook/código, `informe_tecnico.docx` y el video.
