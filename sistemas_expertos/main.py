import streamlit as st

from src.database import init_db
from src.knowledge_base import cargar_sintomas, cargar_trastornos
from src.inference_engine import construir_tabla_reglas, evaluar, hay_alerta_riesgo
from src.report_generator import generar_reporte, RECURSOS_CRISIS, DESCARGO


def main():
    st.set_page_config(
        page_title="Bot de orientación - Sistema Experto",
        page_icon="🩺",
        layout="centered",
    )

    init_db()

    st.title("🩺 Bot de orientación (Sistema Experto - Simulación académica)")

    st.info(
        "Este bot es un proyecto académico de un curso de Inteligencia "
        "Artificial. No es un diagnóstico médico ni psicológico real. "
        "Responde según cómo te has sentido, marcando lo que aplique."
    )

    trastornos = cargar_trastornos()
    sintomas = cargar_sintomas()

    with st.expander("Ver base de conocimiento y reglas de inferencia"):
        st.write(
            f"La base de conocimiento contiene **{len(trastornos)} trastornos** "
            f"y **{len(sintomas)} síntomas únicos**, relacionados entre sí."
        )
        st.write("Tabla de reglas de inferencia (IF ... THEN ...):")
        tabla = construir_tabla_reglas(trastornos)
        filas_md = ["| Regla | Trastorno | Condición (SI) | Conclusión (ENTONCES) |",
                    "|---|---|---|---|"]
        for fila in tabla:
            filas_md.append(
                f"| {fila['regla']} | {fila['trastorno']} | "
                f"{fila['condicion_si']} | {fila['conclusion_entonces']} |"
            )
        st.markdown("\n".join(filas_md))

    st.header("Cuestionario de síntomas")
    st.caption(
        "Marca las casillas que describan lo que has sentido o experimentado "
        "recientemente. Puedes marcar tantas como apliquen."
    )
    respuestas = {}
    for trastorno in trastornos:
        with st.expander(f"{trastorno.categoria} — {trastorno.nombre}"):
            for sintoma, es_nuclear in trastorno.sintomas:
                if sintoma.id in respuestas:
                    # Ya se preguntó este síntoma en otro trastorno (compartido)
                    continue
                etiqueta = sintoma.texto
                if es_nuclear:
                    etiqueta += "  •"
                respuestas[sintoma.id] = st.checkbox(etiqueta, key=f"sintoma_{sintoma.id}")

    st.divider()

    if st.button("Generar diagnóstico preliminar", type="primary"):
        resultados = evaluar(trastornos, respuestas)
        alerta = hay_alerta_riesgo(sintomas, respuestas)

        if alerta:
            st.error(
                "Marcaste una respuesta relacionada con pensamientos de hacerte "
                "daño o de que la vida no vale la pena. Esto es más importante "
                "que cualquier resultado de abajo: por favor busca apoyo ahora."
            )
            st.warning(RECURSOS_CRISIS)

        st.subheader("Resultados preliminares")

        posibles = [r for r in resultados if r.cumple_umbral]
        if posibles:
            for r in posibles:
                st.success(
                    f"**{r.trastorno.nombre}** — {r.coincidencias}/{r.total_sintomas} "
                    f"síntomas coincidentes ({r.porcentaje}%)"
                )
        else:
            st.write(
                "Ningún trastorno de la base de conocimiento alcanzó el umbral "
                "mínimo de coincidencia con lo marcado."
            )

        with st.expander("Ver puntaje de todos los trastornos evaluados"):
            for r in resultados:
                st.write(
                    f"- {r.trastorno.nombre}: {r.coincidencias}/{r.total_sintomas} "
                    f"({r.porcentaje}%) — "
                    f"{'cumple umbral' if r.cumple_umbral else 'no cumple umbral'}"
                )

        st.caption(DESCARGO)

        reporte_texto = generar_reporte(resultados, alerta)
        st.download_button(
            label="Descargar reporte (.txt)",
            data=reporte_texto,
            file_name="reporte_preliminar.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
