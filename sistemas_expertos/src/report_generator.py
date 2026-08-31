"""
Generador de reporte de resultados preliminares.

"""

from datetime import datetime

RECURSOS_CRISIS = (
    "Línea 123 (línea de emergencias, disponible en Medellín y en la mayoría "
    "de ciudades de Colombia marcando 123 desde cualquier operador, las 24 "
    "horas) y Línea 106 'El poder de ser escuchado' (atención, orientación e "
    "intervención en crisis). Si estás fuera de Colombia, busca la línea de "
    "atención en crisis de tu país o acude al servicio de urgencias más cercano."
)

DESCARGO = (
    "Este reporte es el resultado de una simulación académica de un sistema "
    "experto y NO constituye un diagnóstico médico ni psicológico. Los "
    "resultados se basan en una base de conocimiento simplificada, construida "
    "con fines educativos. Cualquier inquietud real sobre salud mental debe "
    "consultarse con un profesional calificado."
)


def generar_reporte(resultados, alerta_riesgo: bool, respuestas_texto=None) -> str:
    lineas = []
    lineas.append("REPORTE PRELIMINAR - SISTEMA EXPERTO (SIMULACIÓN ACADÉMICA)")
    lineas.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append("=" * 70)
    lineas.append("")

    if alerta_riesgo:
        lineas.append("AVISO PRIORITARIO")
        lineas.append("-" * 70)
        lineas.append(
            "Se identificó una respuesta relacionada con pensamientos de "
            "hacerse daño o de que la vida no vale la pena. Independientemente "
            "de los demás resultados, se recomienda buscar apoyo de inmediato:"
        )
        lineas.append(RECURSOS_CRISIS)
        lineas.append("")

    lineas.append("RESULTADOS POR TRASTORNO (ordenados por % de coincidencia)")
    lineas.append("-" * 70)

    posibles = [r for r in resultados if r.cumple_umbral]
    otros = [r for r in resultados if not r.cumple_umbral]

    if posibles:
        lineas.append("Trastornos que cumplen el umbral mínimo de coincidencia:")
        for r in posibles:
            lineas.append(
                f"  - {r.trastorno.nombre} ({r.trastorno.categoria}): "
                f"{r.coincidencias}/{r.total_sintomas} síntomas "
                f"({r.porcentaje}%). Duración orientativa esperada: "
                f"{r.trastorno.duracion_orientativa}."
            )
    else:
        lineas.append(
            "Ningún trastorno de la base de conocimiento alcanzó el umbral "
            "mínimo de coincidencia con los síntomas marcados."
        )

    if otros:
        lineas.append("")
        lineas.append("Otros trastornos evaluados (no alcanzaron el umbral):")
        for r in otros:
            lineas.append(
                f"  - {r.trastorno.nombre}: {r.coincidencias}/{r.total_sintomas} "
                f"síntomas ({r.porcentaje}%)"
            )

    lineas.append("")
    lineas.append("DESCARGO DE RESPONSABILIDAD")
    lineas.append("-" * 70)
    lineas.append(DESCARGO)

    return "\n".join(lineas)
