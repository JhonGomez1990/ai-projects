"""
Motor de inferencia del sistema experto.

"""

from dataclasses import dataclass


@dataclass
class ResultadoTrastorno:
    trastorno: object  # knowledge_base.Trastorno
    coincidencias: int
    total_sintomas: int
    porcentaje: float
    cumple_umbral: bool
    nuclear_presente: bool


def construir_tabla_reglas(trastornos):

    tabla = []
    for i, trastorno in enumerate(trastornos, start=1):
        condicion = f"coincidencias >= {trastorno.min_criterios}"
        if trastorno.requiere_nuclear:
            condicion += " Y al menos 1 síntoma nuclear presente"

        tabla.append(
            {
                "regla": f"R{i}",
                "trastorno": trastorno.nombre,
                "categoria": trastorno.categoria,
                "condicion_si": condicion,
                "conclusion_entonces": f"'{trastorno.nombre}' es posible",
            }
        )
    return tabla


def evaluar(trastornos, respuestas: dict) -> list:

    resultados = []

    for trastorno in trastornos:
        total = len(trastorno.sintomas)
        coincidencias = 0
        nuclear_presente = False

        for sintoma, es_nuclear in trastorno.sintomas:
            if respuestas.get(sintoma.id, False):
                coincidencias += 1
                if es_nuclear:
                    nuclear_presente = True

        cumple_umbral = coincidencias >= trastorno.min_criterios
        if trastorno.requiere_nuclear:
            cumple_umbral = cumple_umbral and nuclear_presente

        porcentaje = round((coincidencias / total) * 100, 1) if total else 0.0

        resultados.append(
            ResultadoTrastorno(
                trastorno=trastorno,
                coincidencias=coincidencias,
                total_sintomas=total,
                porcentaje=porcentaje,
                cumple_umbral=cumple_umbral,
                nuclear_presente=nuclear_presente,
            )
        )

    resultados.sort(key=lambda r: r.porcentaje, reverse=True)
    return resultados


def hay_alerta_riesgo(sintomas, respuestas: dict) -> bool:

    for sintoma in sintomas:
        if sintoma.es_alerta and respuestas.get(sintoma.id, False):
            return True
    return False
