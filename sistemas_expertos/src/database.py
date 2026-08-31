"""
BASE DE DATOS
"""

import sqlite3
from pathlib import Path

from src.seed_data import TRASTORNOS, SINTOMAS_ALERTA

DB_PATH = Path("data") / "expertos.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sintomas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    texto TEXT NOT NULL UNIQUE,
    es_alerta INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trastornos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    categoria TEXT NOT NULL,
    min_criterios INTEGER NOT NULL,
    requiere_nuclear INTEGER NOT NULL DEFAULT 0,
    duracion_orientativa TEXT
);

CREATE TABLE IF NOT EXISTS trastorno_sintoma (
    trastorno_id INTEGER NOT NULL REFERENCES trastornos(id),
    sintoma_id INTEGER NOT NULL REFERENCES sintomas(id),
    es_nuclear INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (trastorno_id, sintoma_id)
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _is_seeded(conn):
    row = conn.execute("SELECT COUNT(*) FROM trastornos").fetchone()
    return row[0] > 0


def init_db():
    """Crea las tablas (si no existen) y puebla la base con la
    base de conocimiento inicial (solo la primera vez)."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()

    if not _is_seeded(conn):
        _seed(conn)

    conn.close()


def _seed(conn):
    for trastorno in TRASTORNOS:
        cur = conn.execute(
            """INSERT INTO trastornos
               (nombre, categoria, min_criterios, requiere_nuclear, duracion_orientativa)
               VALUES (?, ?, ?, ?, ?)""",
            (
                trastorno["nombre"],
                trastorno["categoria"],
                trastorno["min_criterios"],
                int(trastorno["requiere_nuclear"]),
                trastorno["duracion_orientativa"],
            ),
        )
        trastorno_id = cur.lastrowid

        for texto_sintoma, es_nuclear in trastorno["sintomas"]:
            es_alerta = int(texto_sintoma in SINTOMAS_ALERTA)

            cur_sintoma = conn.execute(
                "SELECT id FROM sintomas WHERE texto = ?", (texto_sintoma,)
            ).fetchone()

            if cur_sintoma is None:
                cur2 = conn.execute(
                    "INSERT INTO sintomas (texto, es_alerta) VALUES (?, ?)",
                    (texto_sintoma, es_alerta),
                )
                sintoma_id = cur2.lastrowid
            else:
                sintoma_id = cur_sintoma[0]

            conn.execute(
                """INSERT OR IGNORE INTO trastorno_sintoma
                   (trastorno_id, sintoma_id, es_nuclear) VALUES (?, ?, ?)""",
                (trastorno_id, sintoma_id, int(es_nuclear)),
            )

    conn.commit()


def reset_db():
    """Borra el archivo de base de datos para forzar una nueva siembra.
    Útil si se edita seed_data.py y se quiere reconstruir la base."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()
