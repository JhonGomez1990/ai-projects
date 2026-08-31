"""
Carga la base de conocimiento (trastornos y síntomas).
"""

from dataclasses import dataclass, field
from src.database import get_connection


@dataclass
class Sintoma:
    id: int
    texto: str
    es_alerta: bool


@dataclass
class Trastorno:
    id: int
    nombre: str
    categoria: str
    min_criterios: int
    requiere_nuclear: bool
    duracion_orientativa: str
    sintomas: list = field(default_factory=list)  # list[(Sintoma, es_nuclear)]


def cargar_sintomas():
    """Retorna la lista completa y única de síntomas, ordenada por texto."""
    conn = get_connection()
    filas = conn.execute(
        "SELECT id, texto, es_alerta FROM sintomas ORDER BY texto"
    ).fetchall()
    conn.close()
    return [Sintoma(id=f[0], texto=f[1], es_alerta=bool(f[2])) for f in filas]


def cargar_trastornos():
    """Retorna la lista de trastornos, cada uno con sus síntomas asociados."""
    conn = get_connection()

    filas_trastornos = conn.execute(
        """SELECT id, nombre, categoria, min_criterios,
                  requiere_nuclear, duracion_orientativa
           FROM trastornos ORDER BY categoria, nombre"""
    ).fetchall()

    trastornos = []
    for f in filas_trastornos:
        trastorno = Trastorno(
            id=f[0],
            nombre=f[1],
            categoria=f[2],
            min_criterios=f[3],
            requiere_nuclear=bool(f[4]),
            duracion_orientativa=f[5],
        )

        filas_sintomas = conn.execute(
            """SELECT s.id, s.texto, s.es_alerta, ts.es_nuclear
               FROM trastorno_sintoma ts
               JOIN sintomas s ON s.id = ts.sintoma_id
               WHERE ts.trastorno_id = ?
               ORDER BY ts.es_nuclear DESC, s.texto""",
            (trastorno.id,),
        ).fetchall()

        for sf in filas_sintomas:
            sintoma = Sintoma(id=sf[0], texto=sf[1], es_alerta=bool(sf[2]))
            trastorno.sintomas.append((sintoma, bool(sf[3])))

        trastornos.append(trastorno)

    conn.close()
    return trastornos
