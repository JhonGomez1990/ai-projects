"""
Base de conocimiento del sistema experto.

Estructura de cada trastorno:
- nombre: nombre del trastorno
- categoria: agrupación clínica general
- min_criterios: cantidad mínima de síntomas coincidentes para considerar
  el trastorno como "posible" dentro de la simulación
- requiere_nuclear: si es True, al menos uno de los síntomas marcados como
  "nuclear" para ese trastorno debe estar presente
- duracion_orientativa: texto informativo (no se usa en el cálculo, solo
  se muestra en el reporte como contexto)
- sintomas: lista de (texto_sintoma, es_nuclear)

Los síntomas repetidos entre trastornos comparten el mismo texto para que
el cuestionario no los pregunte dos veces.
"""

SINTOMAS_ALERTA = {
    "Pensamientos recurrentes de que la vida no vale la pena, de hacerse daño o de morir",
}

TRASTORNOS = [
    {
        "nombre": "Trastorno de depresión mayor",
        "categoria": "Trastornos depresivos",
        "min_criterios": 5,
        "requiere_nuclear": True,
        "duracion_orientativa": "La mayor parte del día, casi todos los días, durante al menos 2 semanas",
        "sintomas": [
            ("Ánimo triste, vacío o irritable la mayor parte del día", True),
            ("Pérdida notable de interés o placer en actividades que antes disfrutaba", True),
            ("Cambios importantes en el apetito o el peso (aumento o disminución)", False),
            ("Dificultad para dormir o dormir en exceso", False),
            ("Sensación de agitación o de lentitud notoria para moverse o hablar", False),
            ("Fatiga o pérdida de energía casi todos los días", False),
            ("Sentimientos de inutilidad o culpa excesiva", False),
            ("Dificultad para concentrarse o para tomar decisiones", False),
            ("Pensamientos recurrentes de que la vida no vale la pena, de hacerse daño o de morir", False),
        ],
    },
    {
        "nombre": "Trastorno de ansiedad generalizada",
        "categoria": "Trastornos de ansiedad",
        "min_criterios": 4,
        "requiere_nuclear": True,
        "duracion_orientativa": "La mayoría de los días durante al menos 6 meses",
        "sintomas": [
            ("Preocupación excesiva y difícil de controlar sobre varios temas de la vida diaria", True),
            ("Sensación de inquietud o de estar 'con los nervios de punta'", False),
            ("Fatiga o cansancio con facilidad", False),
            ("Dificultad para concentrarse o quedarse con la mente en blanco", False),
            ("Irritabilidad notoria", False),
            ("Tensión muscular", False),
            ("Dificultad para dormir o sueño poco reparador", False),
        ],
    },
    {
        "nombre": "Trastorno de pánico",
        "categoria": "Trastornos de ansiedad",
        "min_criterios": 4,
        "requiere_nuclear": True,
        "duracion_orientativa": "Episodios recurrentes e inesperados, con preocupación posterior de al menos 1 mes",
        "sintomas": [
            ("Episodios súbitos e intensos de miedo o malestar que alcanzan su pico en minutos", True),
            ("Palpitaciones o sensación de que el corazón late muy fuerte o muy rápido", False),
            ("Sudoración excesiva, temblor o sensación de ahogo durante esos episodios", False),
            ("Sensación de mareo, inestabilidad o desmayo inminente", False),
            ("Miedo a perder el control, 'volverse loco' o morir durante el episodio", False),
            ("Preocupación persistente por tener nuevos episodios o evitar lugares por miedo a que ocurran", False),
        ],
    },
    {
        "nombre": "Trastorno de ansiedad social",
        "categoria": "Trastornos de ansiedad",
        "min_criterios": 3,
        "requiere_nuclear": True,
        "duracion_orientativa": "De manera persistente durante al menos 6 meses",
        "sintomas": [
            ("Miedo o ansiedad marcada en situaciones sociales donde se puede ser observado o juzgado por otros", True),
            ("Miedo a actuar de forma que resulte humillante o vergonzosa", False),
            ("Evitación de situaciones sociales o resistencia a ellas con mucha incomodidad", False),
            ("Ansiedad claramente desproporcionada frente a la situación social real", False),
        ],
    },
    {
        "nombre": "Trastorno obsesivo-compulsivo",
        "categoria": "Trastorno obsesivo-compulsivo y relacionados",
        "min_criterios": 2,
        "requiere_nuclear": True,
        "duracion_orientativa": "Ocupan más de una hora al día o generan malestar significativo",
        "sintomas": [
            ("Pensamientos, imágenes o impulsos repetitivos e indeseados que causan ansiedad (obsesiones)", True),
            ("Necesidad de realizar conductas o rituales repetitivos para reducir la ansiedad (compulsiones)", True),
            ("Dificultad para controlar o detener estos pensamientos o rituales aunque se reconozcan como excesivos", False),
            ("Las obsesiones o compulsiones consumen bastante tiempo del día o interfieren con actividades cotidianas", False),
        ],
    },
    {
        "nombre": "Trastorno de estrés postraumático",
        "categoria": "Trastornos relacionados con trauma y estrés",
        "min_criterios": 4,
        "requiere_nuclear": True,
        "duracion_orientativa": "Más de 1 mes después de un evento traumático",
        "sintomas": [
            ("Haber vivido, presenciado o conocido de cerca un evento traumático o de amenaza grave", True),
            ("Recuerdos intrusivos, pesadillas o 'flashbacks' relacionados con el evento", False),
            ("Evitación de lugares, personas o conversaciones que recuerden el evento", False),
            ("Cambios negativos en pensamientos o estado de ánimo desde el evento (culpa, distanciamiento, visión negativa)", False),
            ("Estado de alerta aumentado: sobresaltos, irritabilidad o dificultad para dormir desde el evento", False),
        ],
    },
    {
        "nombre": "Trastorno por déficit de atención e hiperactividad",
        "categoria": "Trastornos del desarrollo neurológico",
        "min_criterios": 5,
        "requiere_nuclear": False,
        "duracion_orientativa": "Presente desde antes de los 12 años, en más de un entorno, durante al menos 6 meses",
        "sintomas": [
            ("Dificultad frecuente para mantener la atención en tareas o actividades", False),
            ("Se distrae con facilidad ante estímulos externos", False),
            ("Olvida con frecuencia actividades cotidianas o pierde objetos necesarios para tareas", False),
            ("Dificultad para organizar tareas o actividades", False),
            ("Inquietud motora: se mueve en exceso, se levanta o le cuesta permanecer sentado", False),
            ("Habla en exceso o interrumpe con frecuencia a otras personas", False),
            ("Dificultad para esperar turnos o actúa de forma impulsiva", False),
        ],
    },
    {
        "nombre": "Trastorno bipolar (episodio maníaco/hipomaníaco)",
        "categoria": "Trastorno bipolar y trastornos relacionados",
        "min_criterios": 3,
        "requiere_nuclear": True,
        "duracion_orientativa": "Al menos varios días consecutivos, la mayor parte del día",
        "sintomas": [
            ("Periodo notorio de ánimo anormalmente elevado, expansivo o irritable, distinto a lo habitual", True),
            ("Autoestima exageradamente alta o sensación de grandiosidad", False),
            ("Disminución notoria de la necesidad de dormir sin sentirse cansado", False),
            ("Habla más rápido o más de lo habitual, o siente que los pensamientos van muy deprisa", False),
            ("Aumento notorio de la energía o de actividades orientadas a metas", False),
            ("Participación excesiva en actividades de riesgo (gastos, decisiones impulsivas, etc.)", False),
        ],
    },
    {
        "nombre": "Trastorno de insomnio",
        "categoria": "Trastornos del sueño-vigilia",
        "min_criterios": 2,
        "requiere_nuclear": True,
        "duracion_orientativa": "Al menos 3 noches por semana, durante 3 meses o más",
        "sintomas": [
            ("Dificultad persistente para conciliar el sueño, mantenerlo o despertar muy temprano sin poder volver a dormir", True),
            ("Malestar o dificultad notoria durante el día por la falta de sueño (cansancio, mal humor, poca concentración)", False),
            ("Preocupación frecuente por no poder dormir bien", False),
        ],
    },
]
