import sqlite3

conexion = sqlite3.connect("PruebaDB")

cursor = conexion.cursor()

cursor.execute("""
    CREATE TABLE RUBEN (
        PREGUNTA VARCHAR(100),
        RESPUESTA VARCHAR(1000)
    )
""")

datos = [
    ("¿Quién es Rubén Tercero Nueda?","""
Rubén Tercero Nueda es un estudiante del grado en Ingeniería Informática especializado en la rama de Ingeniería de Computadores en la Universidad Politécnica
de Valencia, campus de Alcoy. Actualmente, esta cursando su último año de carrera a falta de entregar su Trabajo Final de Grado.A lo largo de su 
recorrido universitario, obtuvo el Certificado en Estrategia Big Data y Data Analytics para Managers a través de la
propia Universidad y el Certificate in Development of Future technological solutions for health and wellness at LAB University of
Applied Sciences en Lahti(Finlandia).
Su pasión es la inteligencia artificial y el análisis de datos aunque también es un apasionado de las nuevas tecnologias y sus tendencias.
Se considera una persona inspirada que siempre desea marcar la diferencia a través de la colaboración y trabajo con otras personas, afines de gran espiritú.
Motivado para empezar a hacer prácticas de empresa relacionadas en Ingeniería Informática.
"""),
    ("¿Qué nos podrías contar sobre ti?","""
Estoy en mi último año de carrera en el grado de Ingeniería Informática de la Universidad Politécnica de Valencia en el campus de Alcoy y estoy buscando una
oportunidad laboral a través de prácticas de empresa.
A pesar de no tener experiencia previa a lo largo de mi carrera universitaria, como estudiante estuve participando en un grupo de Generación Espontánea
en la universidad acerca del desarrollo de aplicaciones en Realidad Virtual y Realidad Aumentada. También, obtuve dos certificados uno sobre
Estrategia Big Data y Data Analytics para Managers y otro realizado, a través de, la Universidad LAB University of Applied Sciences por Lahti(Finlandia)
sobre el desarrollo de futuras soluciones tecnológicas para la salud y el bienestar. Además, participé en competiciones de Kaggle y Hashcode 
donde mi mejor resultado fue en la competición de G-Research Crypto Forecasting organizado por Kaggle donde quedé en la posición 366 (Top 19%).
"""),
    ("¿Cuáles son tus fortalezas y debilidades?","""
Como debilidad, diría que necesito más tiempo que otras personas para poder aprender a manejar algo nuevo. Todo esto se debe a que me interesa mucho la 
parte teórica de todos los procesos, ya que, siempre he considerado que la teoría es algo fundamental para corregir los problemas más difíciles dentro
de la parte práctica.
Como fortaleza me considero una persona social, motivo por el cual siempre me ha permitido conocer mejor a las personas con las que trabajo
 y eso me ha podido permitir realizar un reparto lógico de tareas a la hora de trabajar en equipo. También, me gusta ampliar mis conocimientos
 en tecnologías nuevas y realizar experimentos que consigan traer propuestas a posibles mejoras del funcionamiento en una empresa.
""")
]

cursor.executemany("INSERT INTO RUBEN VALUES (?,?)", datos)

conexion.commit()

conexion.close()