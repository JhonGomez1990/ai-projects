Para el primer punto, en la carpeta del parcial se encuentra un archivo que contiene las características y tipos de trastornos (Manual DSM). En este aspecto ustedes deberán realizar un sistema experto que recomiende el tipo de trastorno que podría estar padeciendo un usuario que es atendido por un bot. Así las cosas, cada equipo deberá:

* Diseñar una base de datos que represente trastornos y síntomas de manera estructurada.
* Implementar reglas de inferencia que permitan evaluar coincidencias entre síntomas y criterios diagnósticos (Tabla).
* Generar la atención y reporte de resultados preliminares basados en lógica programada (Programa computacional).

Nota de alcance del equipo: la base de conocimiento (trastornos y síntomas) fue redactada por el equipo a partir de las categorías diagnósticas estándar, sin transcribir el contenido protegido por derechos de autor del manual DSM-5, y se usa exclusivamente con fines académicos y de simulación. La aplicación no reemplaza una evaluación clínica profesional.

Usuario (bot)
     ↓
Cuestionario de síntomas
     ↓
Base de datos (trastornos, síntomas, relación trastorno-síntoma)
     ↓
Motor de inferencia (reglas de coincidencia mínima + síntomas nucleares)
     ↓
Puntajes de coincidencia por trastorno
     ↓
Generador de reporte (resultados preliminares + alertas de riesgo)
     ↓
Reporte descargable
