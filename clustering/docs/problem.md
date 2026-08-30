Para el segundo reto, la actividad consiste en desarrollar una aplicación que modifique imágenes utilizando la estrategia de clustering vista en el Bonus de la clase 3. En este caso se requiere de una aplicación que se conecte a la cámara del dispositivo (Webcam), le permita al usuario tomar una foto, y luego escoger la cantidad de grupos que requiere para su objetivo. En este caso:
 
Implementar la captura de imágenes desde la cámara del dispositivo.
Permitir la clasificación manual de imágenes en diferentes clusters.
Organizar, almacenar y descargar la información de forma estructurada.

ameraInput
     ↓
Imagen
     ↓
ClusterProcessor
     ↓
Seleccionar K
     ↓
reshape
     ↓
KMeans
     ↓
Etiquetas
     ↓
Centroides
     ↓
Reconstrucción
     ↓
Imagen procesada