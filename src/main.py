from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import fitz  # PyMuPDF
import openai
import json
import os
from datetime import datetime
import asyncio

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar carpeta estática donde estará index.html y otros archivos
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(f"{STATIC_DIR}/index.html")

# Inicializar la clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Crear directorio de resultados
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def extraer_texto_pdf(file: UploadFile) -> str:
    try:
        contents = file.file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al extraer texto del PDF: {str(e)}")

async def verificar_reputacion_empresa(nombre_empresa: str) -> dict:
    try:
        prompt = f"""
        Con base en conocimiento general, proporciona un análisis breve de la reputación de la empresa: {nombre_empresa}

        Incluye:
        - Puntaje de legitimidad (0-100)
        - Tamaño de la empresa (startup/pequeña/mediana/grande/empresa)
        - Reputación en la industria
        - Problemas conocidos o señales de alerta (si los hay)
        - Resumen breve

        Devuelve un JSON con estos campos exactos.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        return {
            "legitimacy_score": 50,
            "company_size": "desconocido",
            "industry_reputation": "No se pudo verificar",
            "red_flags": [],
            "summary": f"No se pudo verificar la reputación de la empresa: {str(e)}"
        }

async def analizar_cv(texto_cv: str) -> dict:
    prompt = f"""Extrae la siguiente información del CV:
    - Nombre
    - Correo electrónico
    - Teléfono
    - Educación (lista de objetos con título, institución, año)
    - Experiencia laboral (lista de objetos con rol, empresa, fechas, descripción)
    - Habilidades (lista de strings)
    - Idiomas (si se mencionan)
    - Certificaciones (si las hay)
    - Inconsistencias (si las hay, como fechas que se traslapan o patrones sospechosos)

    Devuelve SOLO un JSON válido.

    Texto del CV:
    {texto_cv[:3000]}...
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )

        return json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar el CV: {str(e)}")

async def verificar_empresas_previas(info_cv: dict) -> dict:
    empresas_verificadas = []
    experiencia = info_cv.get("work_experience", [])

    for trabajo in experiencia:
        nombre_empresa = trabajo.get("company", "")
        if nombre_empresa:
            reputacion = await verificar_reputacion_empresa(nombre_empresa)
            empresas_verificadas.append({
                "company": nombre_empresa,
                "role": trabajo.get("role", ""),
                "dates": trabajo.get("dates", ""),
                "reputation": reputacion
            })

    return {
        "companies_checked": len(empresas_verificadas),
        "verification_results": empresas_verificadas,
        "overall_legitimacy_score": sum(c["reputation"]["legitimacy_score"] for c in empresas_verificadas) / len(empresas_verificadas) if empresas_verificadas else 0
    }

async def comparar_cv_con_vacante(info_cv: dict, descripcion_puesto: str, verificacion: dict) -> dict:
    prompt = f"""Compara el perfil del candidato con la descripción del puesto.

    Devuelve:
    - Puntaje de coincidencia general (0-100)
    - Habilidades coincidentes (lista)
    - Habilidades faltantes (lista)
    - Puntaje de relevancia de experiencia (0-100)
    - Puntaje de coincidencia educativa (0-100)
    - Puntaje de reputación de empresas: {verificacion.get('overall_legitimacy_score', 0)}
    - Resumen del ajuste (2-3 frases)
    - Señales de alerta (si hay)
    - Recomendación (altamente_recomendado/recomendado/tal_vez/no_recomendado)

    Perfil del Candidato:
    {json.dumps(info_cv, indent=2)}

    Verificación de Empresas:
    {json.dumps(verificacion, indent=2)}

    Descripción del Puesto:
    {descripcion_puesto[:2000]}...

    Devuelve SOLO un JSON válido.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )

        return json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al comparar CV con puesto: {str(e)}")

@app.post("/subir/")
async def subir_archivos(archivos: List[UploadFile] = File(...), archivo_puesto: UploadFile = File(...)):
    if archivo_puesto.filename.endswith(".pdf"):
        texto_puesto = extraer_texto_pdf(archivo_puesto)
    else:
        try:
            texto_puesto = (await archivo_puesto.read()).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al leer la descripción del puesto: {str(e)}")

    resultados = []

    for archivo in archivos:
        try:
            texto_cv = extraer_texto_pdf(archivo)
            info_cv = await analizar_cv(texto_cv)
            verificacion = await verificar_empresas_previas(info_cv)
            coincidencia = await comparar_cv_con_vacante(info_cv, texto_puesto, verificacion)

            resultado = {
                "nombre_archivo": archivo.filename,
                "timestamp": datetime.now().isoformat(),
                "cv_info": info_cv,
                "verificacion_empresas": verificacion,
                "comparacion_puesto": coincidencia
            }

            nombre_archivo_seguro = archivo.filename.replace(" ", "_").replace(".pdf", "")
            with open(f"{RESULTS_DIR}/{nombre_archivo_seguro}_analisis.json", "w") as f:
                json.dump(resultado, f, indent=2)

            resultados.append(resultado)
        except Exception as e:
            resultados.append({
                "nombre_archivo": archivo.filename,
                "error": str(e)
            })

    return JSONResponse(content={
        "estado": "éxito",
        "archivos_procesados": len(resultados),
        "resultados": resultados
    })

class SolicitudChat(BaseModel):
    message: str
    cv_data: dict
    context: Optional[str] = None

@app.post("/chat/")
async def chat(solicitud: SolicitudChat):
    prompt = f"""
    Eres un asistente de reclutamiento con IA. Con base en este análisis de CV:

    {json.dumps(solicitud.cv_data, indent=2)}

    {f'Contexto adicional: {solicitud.context}' if solicitud.context else ''}

    Responde la siguiente pregunta: {solicitud.message}

    Sé específico y referencia los datos del CV cuando sea posible.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en reclutamiento."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return JSONResponse(content={
            "respuesta": response['choices'][0]['message']['content']
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el chat: {str(e)}")

@app.get("/salud")
async def verificacion_salud():
    return {"estado": "saludable", "timestamp": datetime.now().isoformat()}

@app.get("/resultados")
async def listar_resultados():
    try:
        archivos = os.listdir(RESULTS_DIR)
        return {"resultados": archivos}
    except Exception as e:
        return {"error": str(e), "resultados": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)