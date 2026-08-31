# Sistema Experto - Orientación sobre trastornos (Simulación académica)

Bot de preguntas y respuestas que evalúa síntomas marcados por el usuario
contra una base de conocimiento estructurada de trastornos, y genera un
reporte preliminar de coincidencias mediante reglas de inferencia.

> ⚠️ Este proyecto es una actividad académica del curso de Inteligencia
> Artificial. No es una herramienta de diagnóstico clínico ni reemplaza la
> evaluación de un profesional de la salud mental.

## Arquitectura

```
Usuario (bot)
     ↓
Cuestionario de síntomas (Streamlit)
     ↓
Base de datos SQLite: trastornos, síntomas, relación trastorno-síntoma
     ↓
Motor de inferencia (reglas IF condición THEN trastorno posible)
     ↓
Puntajes de coincidencia + metaregla de alerta de riesgo
     ↓
Generador de reporte (texto descargable)
```

## Base de conocimiento

Incluye 9 trastornos (depresión mayor, ansiedad generalizada, pánico,
ansiedad social, TOC, TEPT, TDAH, trastorno bipolar e insomnio), cada uno
con su lista de síntomas asociados. El texto de los síntomas fue redactado
por el equipo con lenguaje propio, a partir de categorías diagnósticas de
uso clínico general — **no se transcribió contenido del manual DSM-5**,
que está protegido por derechos de autor.

La base se crea y se puebla automáticamente (SQLite, archivo en
`data/expertos.db`) la primera vez que se ejecuta la aplicación. Si editas
`src/seed_data.py` y quieres reconstruir la base desde cero, borra ese
archivo o usa `database.reset_db()`.

## Reglas de inferencia

Cada trastorno tiene asociada una regla:

```
SI (síntomas coincidentes >= mínimo requerido)
   [Y al menos un síntoma "nuclear" está presente, cuando aplica]
ENTONCES el trastorno se marca como "posible", con un % de coincidencia
```

Además existe una metaregla de seguridad, independiente del puntaje: si el
usuario marca el síntoma relacionado con pensamientos de autolesión o de
que la vida no vale la pena, el reporte muestra de inmediato líneas de
atención en crisis, sin importar el resultado del resto de las reglas.

La tabla completa de reglas se puede consultar dentro de la aplicación
("Ver base de conocimiento y reglas de inferencia").

## Uso

### 1. Instalar dependencias

Desde la carpeta `sistemas_expertos/`:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```bash
streamlit run main.py
```

### 3. Abrir la aplicación

Streamlit mostrará una URL local (por ejemplo `http://localhost:8501`).
Ábrela desde el navegador.

### 4. Usar el bot

1. Expande cada categoría y marca los síntomas que apliquen.
2. Presiona **Generar diagnóstico preliminar**.
3. Revisa los trastornos que cumplen el umbral mínimo de coincidencia.
4. Descarga el reporte con el botón **Descargar reporte (.txt)**.
