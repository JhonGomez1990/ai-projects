## Uso de la captura de cámara
### 1. Instalar dependencias

Desde la raíz del proyecto, ejecutar:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

La aplicación debe iniciarse utilizando Streamlit:

```bash
streamlit run main.py
```

También se puede ejecutar con:

```bash
python -m streamlit run main.py
```

> No se recomienda ejecutar la aplicación con `python main.py`, ya que Streamlit necesita crear su propio contexto de ejecución para utilizar componentes como la cámara.

### 3. Abrir la aplicación

Después de ejecutar el comando, Streamlit mostrará una URL local similar a:

```text
http://localhost:8501
```

Abrir esta dirección desde el navegador.

### 4. Permitir acceso a la cámara

Al utilizar la funcionalidad por primera vez, el navegador solicitará permisos para acceder a la cámara del dispositivo.

Seleccionar:

```text
Permitir
```

Si el permiso es rechazado, la aplicación no podrá mostrar la cámara.

### 5. Habilitar la cámara

Dentro de la aplicación:

1. Activar la opción **Enable camera**.
2. Esperar a que se muestre la cámara del dispositivo.
3. Presionar **Take a picture** para realizar la captura.
4. La imagen capturada aparecerá inmediatamente en pantalla.