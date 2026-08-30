import cv2
import numpy as np
from sklearn.cluster import KMeans

class ImageClustering:

    @staticmethod
    def process(image_file, k):

        # Convertir la imagen capturada a un arreglo de NumPy
        image_bytes = np.frombuffer(image_file.getvalue(), dtype=np.uint8)

        Img_2 = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # OpenCV trabaja en BGR, lo convertimos a RGB
        Img_2 = cv2.cvtColor(Img_2, cv2.COLOR_BGR2RGB)

        # Obtener dimensiones de la imagen
        Fl, Cl, Ch = Img_2.shape

        # Ajustando los datos
        dataset = Img_2.reshape((Fl * Cl, Ch))

        X = dataset

        [Muestras, Atributos] = X.shape

        # Crear modelo de clustering
        Modelo_Cluster = KMeans(k)

        Etiquetas = Modelo_Cluster.fit_predict(X)

        Centroides = Modelo_Cluster.cluster_centers_

        # Armando imagen resultante
        Resultado = np.zeros((Fl * Cl, Ch))

        for i in range(Fl * Cl):
            Apuntador = Etiquetas[i]
            Resultado[i, :] = Centroides[Apuntador, :]

        Img_resultado = Resultado.reshape((Fl, Cl, Ch))

        return Img_resultado.astype("uint8")