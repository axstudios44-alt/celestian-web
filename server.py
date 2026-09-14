import os
import sqlite3
import tempfile
import time
import base64
import asyncio
import re
from urllib.parse import quote_plus

import edge_tts
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__, static_folder=".", static_url_path="")

conn = sqlite3.connect("memoria_web.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS mensajes (
    rol TEXT,
    contenido TEXT
)
""")
conn.commit()

SYSTEM_PROMPT = (
    "Eres Celestian, un asistente personal por voz. Respondes siempre en "
    "espanol, de forma breve y natural, como en una conversacion hablada "
    "(evita listas largas o formato de texto, ya que tu respuesta se lee "
    "en voz alta). "
    "Si el usuario te pide abrir una pagina conocida sin buscar nada "
    "especifico (YouTube, Google, Gmail, etc.), responde confirmando y "
    "agrega al final, en una linea aparte, exactamente: "
    "[[ABRIR:https://url-completa-aqui]] con la URL real del sitio. "
    "Si el usuario te pide BUSCAR algo en un sitio (un video, un producto, "
    "informacion, etc.), responde confirmando y agrega al final, en una "
    "linea aparte, exactamente: [[BUSCAR:sitio|texto de busqueda]] donde "
    "sitio es una de estas palabras exactas: youtube, google, amazon, "
    "wikipedia, maps -- y 'texto de busqueda' es lo que se debe buscar, "
    "en texto normal sin codificar. No menciones ni expliques estas "
    "etiquetas, solo agregalas. Si no te piden abrir ni buscar nada, no "
    "agregues ninguna etiqueta."
)

PATRON_ABRIR = re.compile(r"\[\[ABRIR:(https?://[^\]]+)\]\]")
PATRON_BUSCAR = re.compile(r"\[\[BUSCAR:([a-z_]+)\|([^\]]+)\]\]")

PLANTILLAS_BUSQUEDA = {
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "google": "https://www.google.com/search?q={q}",
    "amazon": "https://www.amazon.com/s?k={q}",
    "wikipedia": "https://es.wikipedia.org/w/index.php?search={q}",
    "maps": "https://www.google.com/maps/search/{q}",
}


def extraer_accion(texto):
    """Busca etiquetas de abrir/buscar en el texto y devuelve (texto_limpio, url)."""
    coincidencia = PATRON_BUSCAR.search(texto)
    if coincidencia:
        sitio, consulta = coincidencia.groups()
        plantilla = PLANTILLAS_BUSQUEDA.get(sitio.strip())
        texto_limpio = PATRON_BUSCAR.sub("", texto).strip()
        if plantilla:
            url = plantilla.format(q=quote_plus(consulta.strip()))
            return texto_limpio, url
        return texto_limpio, None

    coincidencia = PATRON_ABRIR.search(texto)
    if coincidencia:
        url = coincidencia.group(1)
        texto_limpio = PATRON_ABRIR.sub("", texto).strip()
        return texto_limpio, url

    return texto, None


def obtener_historial(limite=10):
    cursor.execute(
        "SELECT rol, contenido FROM mensajes ORDER BY rowid DESC LIMIT ?",
        (limite,),
    )
    filas = cursor.fetchall()
    filas.reverse()
    return [{"role": rol, "content": contenido} for rol, contenido in filas]


def guardar_mensaje(rol, contenido):
    cursor.execute(
        "INSERT INTO mensajes (rol, contenido) VALUES (?, ?)",
        (rol, contenido),
    )
    conn.commit()


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


VOCES_DISPONIBLES = [
    {"id": "es-MX-DaliaNeural", "nombre": "Dalia (México, mujer)"},
    {"id": "es-MX-JorgeNeural", "nombre": "Jorge (México, hombre)"},
    {"id": "es-ES-ElviraNeural", "nombre": "Elvira (España, mujer)"},
    {"id": "es-ES-AlvaroNeural", "nombre": "Álvaro (España, hombre)"},
    {"id": "es-AR-ElenaNeural", "nombre": "Elena (Argentina, mujer)"},
    {"id": "es-CO-SalomeNeural", "nombre": "Salomé (Colombia, mujer)"},
    {"id": "es-US-PalomaNeural", "nombre": "Paloma (EEUU, mujer)"},
    {"id": "es-US-AlonsoNeural", "nombre": "Alonso (EEUU, hombre)"},
]


async def _generar_audio_async(texto, voz):
    comunicador = edge_tts.Communicate(texto, voz)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        ruta = tmp.name
    await comunicador.save(ruta)
    with open(ruta, "rb") as f:
        audio_bytes = f.read()
    os.remove(ruta)
    return audio_bytes


def generar_audio_base64(texto, voz):
    try:
        audio_bytes = asyncio.run(_generar_audio_async(texto, voz))
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("Error generando audio TTS:", e)
        return None


@app.route("/api/voces")
def voces_disponibles():
    return jsonify(VOCES_DISPONIBLES)


VISION_MODEL = "qwen/qwen3.6-27b"

PROMPT_VISION = (
    "Eres Celestian, un asistente que observa la pantalla del usuario en "
    "tiempo real para ayudarlo de forma proactiva. Se te muestra una "
    "captura de su pantalla en este momento. Si notas algo realmente util "
    "para comentar (un error visible, algo en lo que claramente necesita "
    "ayuda, informacion relevante que deberia saber), responde con un "
    "comentario breve y natural en espanol, como se diria en voz alta. "
    "Si no hay nada que valga la pena comentar, responde UNICAMENTE con "
    "la palabra: NADA"
)


@app.route("/api/vision", methods=["POST"])
def vision():
    inicio = time.time()
    datos = request.json or {}
    imagen_base64 = datos.get("image")
    voz = datos.get("voice", VOCES_DISPONIBLES[0]["id"])
    if not imagen_base64:
        return jsonify({"error": "Falta la imagen"}), 400

    respuesta = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_VISION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Esto es lo que hay en mi pantalla ahora mismo."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{imagen_base64}"},
                    },
                ],
            },
        ],
        temperature=0.7,
        max_completion_tokens=200,
    )
    texto = respuesta.choices[0].message.content.strip()

    if texto.upper().startswith("NADA"):
        return jsonify({"comentario": None})

    guardar_mensaje("assistant", f"[Sobre tu pantalla] {texto}")
    audio_base64 = generar_audio_base64(texto, voz)
    latencia_ms = int((time.time() - inicio) * 1000)

    return jsonify({
        "comentario": texto,
        "audio_base64": audio_base64,
        "latency_ms": latencia_ms,
    })


def procesar_mensaje(texto_usuario, inicio, voz):
    guardar_mensaje("user", texto_usuario)
    historial = obtener_historial()
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial

    respuesta = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=mensajes,
    )
    texto_respuesta = respuesta.choices[0].message.content
    texto_respuesta, url_a_abrir = extraer_accion(texto_respuesta)

    guardar_mensaje("assistant", texto_respuesta)

    latencia_ms = int((time.time() - inicio) * 1000)
    audio_base64 = generar_audio_base64(texto_respuesta, voz)

    return {
        "transcript": texto_usuario,
        "reply": texto_respuesta,
        "latency_ms": latencia_ms,
        "audio_base64": audio_base64,
        "open_url": url_a_abrir,
    }


@app.route("/api/talk", methods=["POST"])
def talk():
    inicio = time.time()
    audio = request.files["audio"]
    voz = request.form.get("voice", VOCES_DISPONIBLES[0]["id"])

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio.save(tmp.name)
        ruta_temp = tmp.name

    with open(ruta_temp, "rb") as f:
        transcripcion = groq_client.audio.transcriptions.create(
            file=("audio.webm", f.read()),
            model="whisper-large-v3-turbo",
        )
    os.remove(ruta_temp)

    texto_usuario = transcripcion.text.strip()
    if not texto_usuario:
        return jsonify({"error": "No se entendio el audio"}), 400

    return jsonify(procesar_mensaje(texto_usuario, inicio, voz))


@app.route("/api/text", methods=["POST"])
def text():
    inicio = time.time()
    datos = request.json or {}
    texto_usuario = datos.get("text", "").strip()
    voz = datos.get("voice", VOCES_DISPONIBLES[0]["id"])
    if not texto_usuario:
        return jsonify({"error": "Mensaje vacio"}), 400

    return jsonify(procesar_mensaje(texto_usuario, inicio, voz))


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto)
