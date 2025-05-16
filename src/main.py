import os
import json
import uuid
from typing import List, Dict, Any

import openai
import wikipedia
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdfminer.high_level import extract_text as extract_pdf_text
from PIL import Image
import pytesseract

# Configurar la clave de API de OpenAI
ENV_API_KEY = "OPENAI_API_KEY"
if ENV_API_KEY not in os.environ:
    raise RuntimeError(f"Debe establecer la variable de entorno {ENV_API_KEY}")
openai.api_key = os.getenv(ENV_API_KEY)

# Modelo de OpenAI a utilizar (GPT-3.5 Turbo)
MODEL = "gpt-3.5-turbo"

# Descripción del puesto editable por el usuario (puede moverse al frontend si se desea)
descripcion_puesto: Dict[str, Any] = {
    "titulo": "Ingeniero de Software Senior",
    "responsabilidades": [
        "Diseñar e implementar sistemas backend escalables.",
        "Colaborar con equipos multifuncionales.",
        "Mantener estándares de calidad y pruebas automatizadas."
    ],
    "requisitos": [
        "5+ años de experiencia en Python y Django.",
        "Conocimiento de Docker y Kubernetes.",
        "Nivel avanzado de inglés."
    ]
}

# Inicializar la aplicación FastAPI
app = FastAPI(title="TalentAI", version="1.0")

# Configurar CORS para permitir acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario en memoria para almacenar el contexto de chat por sesión
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# ------------------- Funciones de Procesamiento -------------------

def extraer_texto_pdf(path: str) -> str:
    """Extrae texto de un archivo PDF usando pdfminer."""
    return extract_pdf_text(path)

def extraer_texto_imagen(path: str) -> str:
    """Extrae texto de una imagen utilizando OCR (pytesseract)."""
    image = Image.open(path)
    return pytesseract.image_to_string(image)

def parsear_cv(texto: str) -> Dict[str, Any]:
    """
    Envía el texto plano de un CV a OpenAI para extraer información estructurada:
      - work_experience: lista de {title, company, start_date, end_date}
      - education: lista de {degree, institution, start_date, end_date}
      - languages: lista de {language, proficiency}
      - skills: lista de strings
    """
    system_msg = (
        "Eres un experto en RRHH y formateas CVs. "
        "Recibe texto de un CV y extrae en JSON las claves: "
        "\u2022 work_experience: lista de objetos con campos title, company, start_date, end_date. "
        "\u2022 education: lista de objetos con fields degree, institution, start_date, end_date. "
        "\u2022 languages: lista de objetos language y proficiency. "
        "\u2022 skills: lista de cadenas."
    )
    user_msg = f"Texto del CV:\n{texto}"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=1500
    )
    return json.loads(response.choices[0].message.content)

def verificar_empresa(nombre: str) -> bool:
    """Consulta Wikipedia para verificar si una empresa existe."""
    try:
        wikipedia.page(nombre)
        return True
    except Exception:
        return False

def verificar_empresas(empresas: List[str]) -> Dict[str, bool]:
    """Verifica una lista de nombres de empresas."""
    return {e: verificar_empresa(e) for e in empresas if e}

def calcular_compatibilidad(cv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara un perfil de CV con la descripción del puesto y devuelve:
      - compatibility_percentage (int)
      - strengths (fortalezas)
      - gaps (brechas)
    """
    system_msg = (
        "Eres un especialista de selección de personal. "
        "Comparas un perfil de candidato con una descripción de puesto y devuelves en JSON: "
        "compatibility_percentage (0-100), strengths (fortalezas), gaps (brechas)."
    )
    user_msg = (
        f"Perfil del candidato (JSON): {json.dumps(cv, ensure_ascii=False)}\n"
        f"Descripción del puesto (JSON): {json.dumps(descripcion_puesto, ensure_ascii=False)}"
    )
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=800
    )
    return json.loads(response.choices[0].message.content)

# ------------------- Endpoints de API -------------------

@app.post("/procesar_cvs")
async def procesar_cvs(cvs: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    """
    Procesa uno o varios CVs cargados:
      - Extrae texto del archivo
      - Llama a OpenAI para extraer datos estructurados
      - Verifica las empresas en Wikipedia
      - Compara con la descripción del puesto
      - Inicia una sesión de chat por CV
    """
    resultados = []
    for archivo in cvs:
        contenido = await archivo.read()
        ext = os.path.splitext(archivo.filename)[1].lower()
        tmp_path = f"/tmp/{uuid.uuid4().hex}{ext}"
        with open(tmp_path, 'wb') as f:
            f.write(contenido)

        if ext == ".pdf":
            texto = extraer_texto_pdf(tmp_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            texto = extraer_texto_imagen(tmp_path)
        else:
            texto = contenido.decode('utf-8')
        os.remove(tmp_path)

        cv_info = parsear_cv(texto)
        empresas = [e.get('company') for e in cv_info.get('work_experience', [])]
        cv_info['verificacion_empresas'] = verificar_empresas(empresas)
        compat = calcular_compatibilidad(cv_info)

        session_id = str(uuid.uuid4())
        # Iniciar conversación de chat con el contexto del CV y el análisis
        chat_sessions[session_id] = [
            {"role": "system", "content": "Eres un asistente de talento humano."},
            {"role": "system", "content": json.dumps({"cv": cv_info, "job": descripcion_puesto, "compat": compat}, ensure_ascii=False)}
        ]

        resultados.append({
            "session_id": session_id,
            "filename": archivo.filename,
            "cv_info": cv_info,
            "compatibilidad": compat
        })
    return resultados

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Endpoint para continuar conversación con un CV previamente analizado.
    Utiliza el contexto almacenado para responder en base a ese perfil.
    """
    sesiones = chat_sessions.get(request.session_id)
    if not sesiones:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    sesiones.append({"role": "user", "content": request.message})
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=sesiones,
        temperature=0.7,
        max_tokens=500
    )
    reply = response.choices[0].message.content
    sesiones.append({"role": "assistant", "content": reply})
    return {"reply": reply}

# ------------------- Ejecutar como aplicación -------------------

if __name__ == "__main__":
    import uvicorn
    # Ejecutar FastAPI en modo recarga automática para desarrollo
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)