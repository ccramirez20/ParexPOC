import os
import json
import requests
import openai
import fitz  # PyMuPDF para trabajar con PDFs
from typing import List, Dict, Optional, Any, Union
import logging
from datetime import datetime
import re
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pydantic import BaseModel
from pathlib import Path

# Configuración de logging para un mejor seguimiento de errores y procesos
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI(title="CV Parser API", description="API para análisis de currículums vitae")

# Configurar directorios
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios si no existen
for dir_path in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Configuración de templates y archivos estáticos
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configuración de la API de OpenAI
openai.api_key = "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"

# URL del servicio OCR
OCR_URL = "https://api.ocr.space/parse/image"
API_KEY = "K81778047988957"  # Clave de API para OCR.space

# Descripción del trabajo para el POC
DESCRIPCION_TRABAJO = """
Buscamos un Desarrollador Full Stack con experiencia en:
- Mínimo 3 años de experiencia en desarrollo web
- Dominio de Python y JavaScript
- Experiencia con frameworks como React, Angular o Vue.js
- Conocimientos de bases de datos SQL y NoSQL
- Experiencia con metodologías ágiles (Scrum, Kanban)
- Familiaridad con servicios cloud (AWS, Azure o Google Cloud)
- Git y CI/CD pipelines
- API REST y GraphQL
- Capacidad de trabajar en equipo
- Inglés intermedio o avanzado

Responsabilidades:
- Desarrollar y mantener aplicaciones web
- Colaborar con el equipo de diseño y producto
- Escribir código limpio y documentado
- Participar en code reviews
- Resolver problemas técnicos complejos
"""

# Modelos Pydantic para la API
class JobDescription(BaseModel):
    description: str = DESCRIPCION_TRABAJO
    
class CVAnalysisResult(BaseModel):
    analysis_id: str
    status: str
    progress: int = 0
    result: Optional[Dict] = None
    
class AnalysisStatus(BaseModel):
    status: str
    progress: int
    message: str

class AnalysisInput(BaseModel):
    cv_id: str
    content: str  # resumen o análisis del CV

class ChatRequest(BaseModel):
    message: str

class ChatInput(BaseModel):
    cv_id: str
    message: str

class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class ChatHistoryResponse(BaseModel):
    history: List[ChatMessage]

# Estado global para seguimiento de tareas en background
analysis_tasks = {}

# Memoria en memoria
cv_analyses = {}  # id_cv -> análisis
chat_histories = {}  # id_cv -> historial (lista de mensajes)

# Función para extraer texto de un PDF usando PyMuPDF (fitz)
def extraer_texto_pdf(ruta_archivo: str) -> str:
    try:
        doc = fitz.open(ruta_archivo)
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        return texto.strip()
    except Exception as e:
        logger.error(f"Error al procesar el PDF {ruta_archivo}: {e}")
        return ""

# Función para extraer texto usando OCR cuando el PDF es una imagen escaneada
def extraer_texto_ocr(ruta_archivo: str) -> str:
    try:
        with open(ruta_archivo, 'rb') as f:
            respuesta = requests.post(
                OCR_URL,
                files={'file': f},
                data={
                    'language': 'spa',
                    'isOverlayRequired': False,
                    'apikey': API_KEY
                },
                timeout=120  # Aumentamos el timeout para archivos grandes
            )
            
        if respuesta.status_code != 200:
            logger.error(f"Error en la solicitud OCR. Código: {respuesta.status_code}")
            return ""
            
        resultado = respuesta.json()
        
        if resultado.get("IsErroredOnProcessing"):
            logger.error(f"Error en OCR: {resultado.get('ErrorMessage')}")
            return ""
            
        textos = [item["ParsedText"] for item in resultado.get("ParsedResults", [])]
        return "\n".join(textos)
        
    except Exception as e:
        logger.error(f"Error en el proceso OCR para {ruta_archivo}: {e}")
        return ""

# Función para decidir si un CV es texto o imagen y extraer el texto correspondiente
def extraer_texto(ruta_archivo: str) -> str:
    # Primero intentamos extraer texto del PDF
    texto = extraer_texto_pdf(ruta_archivo)
    
    # Comprobamos si el texto extraído es significativo
    if texto and len(texto) > 100:  # Asumimos que un CV real tendrá más de 100 caracteres
        logger.info(f"Texto extraído correctamente de {ruta_archivo}")
        return texto
    else:
        # Si no encontramos texto suficiente, intentamos usar OCR
        logger.info(f"El archivo {ruta_archivo} parece ser una imagen o PDF escaneado. Usando OCR.")
        return extraer_texto_ocr(ruta_archivo)

# Función para interactuar con OpenAI y extraer información estructurada
def analizar_cv_con_llm(texto: str) -> Dict:
    """
    Envía el texto del CV a OpenAI y obtiene información estructurada.
    """
    if not texto or len(texto) < 50:
        logger.warning("El texto proporcionado es demasiado corto para un análisis efectivo")
        return {
            "error": "Texto insuficiente para analizar",
            "texto_original_longitud": len(texto) if texto else 0
        }
    
    try:
        # Creamos un prompt detallado para extraer la información de manera estructurada
        prompt = """
        Analiza el siguiente CV y extrae la información en formato JSON según estas categorías:
        
        1. nombre: Nombre completo de la persona
        2. edad: Edad de la persona o fecha de nacimiento (si está disponible)
        3. estudios: Lista de objetos con los siguientes campos:
           - titulo: Título o grado obtenido
           - institucion: Nombre de la universidad o institución
           - ubicacion: Ciudad o país
           - periodo: Período de estudio (formato: "YYYY - YYYY" o "YYYY - presente")
           - descripcion: Información adicional relevante
        4. experiencia_previa: Lista de objetos con los siguientes campos:
           - puesto: Cargo o posición ocupada
           - empresa: Nombre de la empresa
           - ubicacion: Ciudad o país
           - periodo: Duración (formato: "YYYY - YYYY" o "YYYY - presente")
           - descripcion: Responsabilidades principales o logros
        5. habilidades: Lista de habilidades técnicas y blandas (strings)
        6. cursos_certificaciones: Lista de objetos con:
           - nombre: Nombre del curso o certificación
           - institucion: Entidad que lo emitió
           - fecha: Año de obtención
           - descripcion: Detalles adicionales si están disponibles
           
        Si alguna información no está presente en el CV, incluye el campo pero con valor null.
        IMPORTANTE: Asegúrate de que la respuesta sea un JSON válido. No incluyas explicaciones adicionales.
        
        CV a analizar:
        """
        prompt += texto
        
        # Realizamos la llamada a la API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # O el modelo que estés utilizando
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en extraer información estructurada de currículums vitae."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Temperatura baja para obtener respuestas más consistentes
        )
        
        # Extraemos la respuesta del modelo
        resultado_text = response.choices[0].message.content.strip()
        
        # Limpiamos la respuesta en caso de que incluya marcadores de código JSON
        if resultado_text.startswith("```json"):
            resultado_text = resultado_text[7:]
        if resultado_text.endswith("```"):
            resultado_text = resultado_text[:-3]
        
        resultado_text = resultado_text.strip()
        
        # Convertimos la respuesta a un diccionario Python
        try:
            resultado = json.loads(resultado_text)
            logger.info("Análisis del CV completado correctamente con LLM")
            return resultado
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de la respuesta LLM: {e}")
            logger.error(f"Texto recibido: {resultado_text[:200]}...")
            
            # Intentar una limpieza más agresiva y reintento
            try:
                # Eliminar cualquier carácter no ASCII
                clean_text = re.sub(r'[^\x00-\x7F]+', '', resultado_text)
                resultado = json.loads(clean_text)
                logger.info("Análisis del CV completado tras limpieza de texto")
                return resultado
            except:
                return {"error": "Formato JSON inválido en la respuesta", "texto_parcial": resultado_text[:500]}
            
    except Exception as e:
        logger.error(f"Error al procesar con OpenAI: {str(e)}")
        return {"error": f"Error en la API de OpenAI: {str(e)}"}

# Función para evaluar la compatibilidad con la descripción de trabajo
def evaluar_compatibilidad(cv_info: Dict, descripcion_trabajo: str = DESCRIPCION_TRABAJO) -> Dict:
    """
    Evalúa qué tan bien se ajusta un candidato a la descripción del trabajo.
    """
    try:
        # Verificamos si el CV tiene información suficiente
        if "error" in cv_info or not cv_info.get("nombre"):
            return {
                "compatibilidad_general": 0,
                "recomendacion": "No evaluado",
                "justificacion": "Información insuficiente en el CV"
            }
            
        # Preparamos el prompt para la evaluación
        prompt = f"""
        Analiza el siguiente perfil de candidato y evalúa su compatibilidad con la descripción del trabajo.
        
        Perfil del candidato:
        {json.dumps(cv_info, indent=2, ensure_ascii=False)}
        
        Descripción del trabajo:
        {descripcion_trabajo}
        
        Evalúa los siguientes aspectos y proporciona una respuesta en formato JSON:
        1. compatibilidad_general: Puntuación del 0 al 100
        2. experiencia_relevante: Puntuación del 0 al 100 y explicación
        3. habilidades_coincidentes: Lista de habilidades que coinciden con los requisitos
        4. habilidades_faltantes: Lista de habilidades requeridas que no se encontraron
        5. anos_experiencia: Años de experiencia relevante detectados
        6. nivel_idioma: Nivel de inglés si se menciona
        7. fortalezas: Lista de puntos fuertes del candidato para el puesto
        8. areas_mejora: Lista de áreas donde el candidato podría mejorar
        9. recomendacion: "Altamente recomendado", "Recomendado", "Con reservas", "No recomendado"
        10. justificacion: Explicación detallada de la evaluación
        
        Asegúrate de devolver un JSON válido.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en recursos humanos evaluando candidatos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        resultado_text = response.choices[0].message.content.strip()
        
        # Limpiamos la respuesta
        if resultado_text.startswith("```json"):
            resultado_text = resultado_text[7:]
        if resultado_text.endswith("```"):
            resultado_text = resultado_text[:-3]
        
        try:
            evaluacion = json.loads(resultado_text.strip())
            logger.info(f"Evaluación de compatibilidad completada para {cv_info.get('nombre', 'candidato desconocido')}")
            return evaluacion
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de evaluación: {e}")
            # Devolvemos un diccionario básico en caso de error
            return {
                "compatibilidad_general": 50,
                "recomendacion": "No evaluado completamente",
                "justificacion": "Error al procesar la evaluación. Se requiere revisión manual."
            }
        
    except Exception as e:
        logger.error(f"Error al evaluar compatibilidad: {e}")
        return {
            "error": f"Error en la evaluación: {str(e)}",
            "compatibilidad_general": 0,
            "recomendacion": "No evaluado"
        }

# Función para verificar la existencia de las empresas con manejo de errores mejorado
def verificar_empresas(cv_info: Dict) -> Dict:
    """
    Verifica la existencia y reputación de las empresas mencionadas en el CV.
    """
    empresas_verificadas = []
    
    if not cv_info.get("experiencia_previa"):
        return {"empresas_verificadas": [], "mensaje": "No se encontraron empresas para verificar"}
    
    for experiencia in cv_info.get("experiencia_previa", []):
        empresa = experiencia.get("empresa", "")
        if not empresa:
            continue
            
        logger.info(f"Verificando empresa: {empresa}")
        
        try:
            # Usando la Wikipedia API para verificación básica
            empresa_normalizada = empresa.replace(' ', '_')
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{empresa_normalizada}"
            headers = {'User-Agent': 'CVParser/1.0'}
            
            try:
                wiki_response = requests.get(wiki_url, headers=headers, timeout=10)
                wiki_data = wiki_response.json() if wiki_response.status_code == 200 else {}
            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Error al consultar Wikipedia para {empresa}: {e}")
                wiki_data = {}
            
            empresa_info = {
                "nombre": empresa,
                "verificada": False,
                "fuente": "No verificada",
                "descripcion": "No se encontró información",
                "confiabilidad": "Baja"
            }
            
            if wiki_data.get("type") == "standard":
                empresa_info.update({
                    "verificada": True,
                    "fuente": "Wikipedia",
                    "descripcion": wiki_data.get("extract", "")[:200] + "...",
                    "confiabilidad": "Alta"
                })
            else:
                # Como respaldo, marcamos como "verificación pendiente"
                empresa_info.update({
                    "verificada": False,
                    "fuente": "Búsqueda manual requerida",
                    "descripcion": "Requiere verificación manual",
                    "confiabilidad": "Pendiente"
                })
            
            empresas_verificadas.append(empresa_info)
            
            # Pequeña pausa para evitar throttling
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error al verificar empresa {empresa}: {e}")
            empresas_verificadas.append({
                "nombre": empresa,
                "verificada": False,
                "fuente": "Error",
                "descripcion": f"Error al verificar: {str(e)}",
                "confiabilidad": "Desconocida"
            })
    
    # Calculamos un score de confiabilidad general
    empresas_verificadas_count = sum(1 for e in empresas_verificadas if e["verificada"])
    total_empresas = len(empresas_verificadas)
    confiabilidad_general = (empresas_verificadas_count / total_empresas * 100) if total_empresas > 0 else 0
    
    return {
        "empresas_verificadas": empresas_verificadas,
        "total_empresas": total_empresas,
        "empresas_confirmadas": empresas_verificadas_count,
        "confiabilidad_general": confiabilidad_general,
        "mensaje": f"Se verificaron {empresas_verificadas_count} de {total_empresas} empresas"
    }

# Función para analizar un documento completo con manejo de errores mejorado
def analizar_documento(ruta_archivo: str, descripcion_trabajo: str = DESCRIPCION_TRABAJO) -> Dict:
    """
    Proceso completo de análisis de un CV: extracción de texto y análisis estructurado.
    """
    logger.info(f"Iniciando análisis del documento: {ruta_archivo}")
    
    try:
        # Extraemos el texto del documento (PDF o imagen)
        texto = extraer_texto(ruta_archivo)
        
        if not texto:
            logger.warning(f"No se pudo extraer texto de {ruta_archivo}")
            return {
                "archivo": os.path.basename(ruta_archivo),
                "error": "No se pudo extraer texto del documento"
            }
        
        # Analizamos el texto con el LLM para extraer información estructurada
        resultado = analizar_cv_con_llm(texto)
        
        # Agregamos el nombre del archivo al resultado
        resultado["archivo"] = os.path.basename(ruta_archivo)
        
        # Evaluamos la compatibilidad con el trabajo solo si no hay errores
        if "error" not in resultado:
            logger.info("Evaluando compatibilidad del candidato...")
            evaluacion = evaluar_compatibilidad(resultado, descripcion_trabajo)
            resultado["evaluacion_compatibilidad"] = evaluacion
            
            logger.info("Verificando empresas del candidato...")
            verificacion_empresas = verificar_empresas(resultado)
            resultado["verificacion_empresas"] = verificacion_empresas
        
        return resultado
        
    except Exception as e:
        logger.error(f"Error general en análisis de documento: {e}")
        return {
            "archivo": os.path.basename(ruta_archivo) if ruta_archivo else "desconocido",
            "error": f"Error en el análisis: {str(e)}"
        }

# Función para generar un resumen ejecutivo
def generar_resumen_ejecutivo(resultados: List[Dict]) -> str:
    """
    Genera un resumen ejecutivo de todos los candidatos procesados.
    """
    resumen = "=== RESUMEN EJECUTIVO DE CANDIDATOS ===\n\n"
    candidatos_recomendados = []
    candidatos_con_reservas = []
    candidatos_no_recomendados = []
    candidatos_con_error = []
    
    for resultado in resultados:
        if "error" in resultado:
            candidatos_con_error.append({
                "nombre": resultado.get("nombre", "Desconocido"),
                "archivo": resultado.get("archivo", ""),
                "error": resultado.get("error", "Error no especificado")
            })
            continue
            
        nombre = resultado.get("nombre", "Desconocido")
        evaluacion = resultado.get("evaluacion_compatibilidad", {})
        recomendacion = evaluacion.get("recomendacion", "No evaluado")
        puntuacion = evaluacion.get("compatibilidad_general", 0)
        
        candidato_resumen = {
            "nombre": nombre,
            "puntuacion": puntuacion,
            "recomendacion": recomendacion,
            "justificacion": evaluacion.get("justificacion", "Sin evaluación")
        }
        
        if recomendacion == "Altamente recomendado":
            candidatos_recomendados.append(candidato_resumen)
        elif recomendacion in ["Recomendado", "Con reservas"]:
            candidatos_con_reservas.append(candidato_resumen)
        else:
            candidatos_no_recomendados.append(candidato_resumen)
    
    # Ordenar por puntuación
    candidatos_recomendados.sort(key=lambda x: x["puntuacion"], reverse=True)
    candidatos_con_reservas.sort(key=lambda x: x["puntuacion"], reverse=True)
    
    resumen += f"Total de CVs procesados: {len(resultados)}\n"
    resumen += f"Candidatos altamente recomendados: {len(candidatos_recomendados)}\n"
    resumen += f"Candidatos con reservas: {len(candidatos_con_reservas)}\n"
    resumen += f"Candidatos no recomendados: {len(candidatos_no_recomendados)}\n"
    resumen += f"CVs con errores: {len(candidatos_con_error)}\n\n"
    
    if candidatos_recomendados:
        resumen += "CANDIDATOS ALTAMENTE RECOMENDADOS:\n"
        for candidato in candidatos_recomendados:
            resumen += f"- {candidato['nombre']} (Puntuación: {candidato['puntuacion']}%)\n"
            resumen += f"  {candidato['justificacion']}\n\n"
    
    if candidatos_con_reservas:
        resumen += "CANDIDATOS CON RESERVAS:\n"
        for candidato in candidatos_con_reservas:
            resumen += f"- {candidato['nombre']} (Puntuación: {candidato['puntuacion']}%)\n"
            resumen += f"  {candidato['justificacion']}\n\n"
    
    if candidatos_con_error:
        resumen += "CVs CON ERRORES DE PROCESAMIENTO:\n"
        for candidato in candidatos_con_error:
            resumen += f"- Archivo: {candidato['archivo']}\n"
            resumen += f"  Error: {candidato['error']}\n\n"
    
    return resumen

# Función para validar y enriquecer los resultados
def validar_y_enriquecer_resultados(resultados: List[Dict]) -> List[Dict]:
    """
    Realiza validaciones y mejoras adicionales a los resultados obtenidos.
    """
    resultados_validados = []
    
    for resultado in resultados:
        # Verificamos si hay errores críticos
        if "error" in resultado and not resultado.get("nombre"):
            resultados_validados.append(resultado)
            continue
        
        # Aseguramos que existan todas las claves necesarias
        campos_requeridos = ["nombre", "edad", "estudios", "experiencia_previa", 
                            "habilidades", "cursos_certificaciones"]
        
        for campo in campos_requeridos:
            if campo not in resultado:
                resultado[campo] = None
        
        # Realizamos validaciones específicas
        if resultado["nombre"] and isinstance(resultado["nombre"], str):
            # Formateamos el nombre (primera letra en mayúscula de cada palabra)
            resultado["nombre"] = ' '.join(word.capitalize() for word in resultado["nombre"].split())
        
        # Conversiones de tipos según sea necesario
        if not isinstance(resultado.get("habilidades", []), list):
            resultado["habilidades"] = []
            
        if not isinstance(resultado.get("estudios", []), list):
            resultado["estudios"] = []
            
        if not isinstance(resultado.get("experiencia_previa", []), list):
            resultado["experiencia_previa"] = []
            
        if not isinstance(resultado.get("cursos_certificaciones", []), list):
            resultado["cursos_certificaciones"] = []
        
        # Agregamos un timestamp del procesamiento
        resultado["fecha_procesamiento"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        resultados_validados.append(resultado)
    
    return resultados_validados

# Función para procesar un CV en background con mejor manejo de errores
async def procesar_cv_background(file_path: str, analysis_id: str, job_description: str = DESCRIPCION_TRABAJO):
    """
    Procesa un CV en segundo plano y actualiza el estado del análisis.
    """
    try:
        # Actualizar estado a "procesando"
        analysis_tasks[analysis_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Extrayendo texto del CV..."
        }
        
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        # Analizar el documento
        analysis_tasks[analysis_id]["progress"] = 30
        analysis_tasks[analysis_id]["message"] = "Analizando contenido del CV..."
        resultado = analizar_documento(file_path, job_description)
        
        # Actualizar progreso
        analysis_tasks[analysis_id]["progress"] = 50
        analysis_tasks[analysis_id]["message"] = "Validando resultados..."
        
        # Validar y enriquecer el resultado
        resultado_validado = validar_y_enriquecer_resultados([resultado])[0]
        
        # Guardar el resultado en un archivo JSON
        analysis_tasks[analysis_id]["progress"] = 80
        analysis_tasks[analysis_id]["message"] = "Guardando resultados..."
        
        result_file = RESULTS_DIR / f"{analysis_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(resultado_validado, f, indent=4, ensure_ascii=False)
        
        # Actualizar estado a "completado"
        analysis_tasks[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Análisis completado",
            "result": resultado_validado
        }
        
        logger.info(f"Análisis {analysis_id} completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el análisis {analysis_id}: {e}")
        # Actualizar estado a "error"
        analysis_tasks[analysis_id] = {
            "status": "error",
            "progress": 100,
            "message": f"Error: {str(e)}"
        }

# Procesar múltiples CVs en background
async def procesar_cvs_batch_background(file_paths: List[str], batch_id: str, job_description: str = DESCRIPCION_TRABAJO):
    """
    Procesa un lote de CVs en segundo plano.
    """
    try:
        total_files = len(file_paths)
        resultados = []
        
        analysis_tasks[batch_id] = {
            "status": "processing",
            "progress": 0,
            "message": f"Iniciando procesamiento de {total_files} CVs...",
            "total_files": total_files,
            "processed_files": 0
        }
        
        for i, file_path in enumerate(file_paths, 1):
            # Actualizar progreso
            progress = int((i-1) / total_files * 100)
            analysis_tasks[batch_id].update({
                "progress": progress,
                "message": f"Procesando CV {i}/{total_files}: {os.path.basename(file_path)}",
                "processed_files": i-1
            })
            
            # Analizar el documento
            resultado = analizar_documento(file_path, job_description)
            resultados.append(resultado)
            
            # Actualizar contador de procesados
            analysis_tasks[batch_id]["processed_files"] = i
        
        # Validar y enriquecer los resultados
        resultados_validados = validar_y_enriquecer_resultados(resultados)
        
        # Generar resumen ejecutivo
        resumen = generar_resumen_ejecutivo(resultados_validados)
        
        # Guardar los resultados
        result_file = RESULTS_DIR / f"{batch_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(resultados_validados, f, indent=4, ensure_ascii=False)
        
        # Guardar el resumen
        summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(resumen)
        
        # Actualizar estado a "completado"
        analysis_tasks[batch_id] = {
            "status": "completed",
            "progress": 100,
            "message": f"Análisis de {total_files} CVs completado",
            "total_files": total_files,
            "processed_files": total_files,
            "results": resultados_validados,
            "summary": resumen
        }
        
        logger.info(f"Procesamiento por lotes {batch_id} completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el procesamiento por lotes {batch_id}: {e}")
        analysis_tasks[batch_id] = {
            "status": "error",
            "progress": 100,
            "message": f"Error: {str(e)}"
        }

# Guardar análisis
@app.post("/analysis/")
def save_analysis(data: AnalysisInput):
    cv_analyses[data.cv_id] = data.content
    chat_histories[data.cv_id] = [
        {"role": "system", "content": f"Eres un asistente de RRHH. Este es el análisis del CV del candidato: {data.content}"}
    ]
    return {"status": "análisis guardado"}

# Enviar mensaje al chat
@app.post("/chat/", response_model=ChatMessage)
def chat_with_cv_context(data: ChatInput):
    if data.cv_id not in chat_histories:
        raise HTTPException(status_code=404, detail="CV no encontrado o sin análisis")

    chat_histories[data.cv_id].append({"role": "user", "content": data.message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_histories[data.cv_id]
    )

    reply = response["choices"][0]["message"]
    chat_histories[data.cv_id].append(reply)

    return reply

# Ver historial del chat
@app.get("/chat/{cv_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(cv_id: str):
    if cv_id not in chat_histories:
        raise HTTPException(status_code=404, detail="CV no encontrado")

    return {"history": chat_histories[cv_id]}

# Rutas de la API FastAPI
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Página principal de la aplicación.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload-cv/")
async def upload_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...), job_description: str = Form(DESCRIPCION_TRABAJO)):
    """
    Endpoint para subir y analizar un CV.
    """
    try:
        # Generar un ID único para este análisis
        analysis_id = f"cv_{int(time.time())}_{file.filename.replace(' ', '_')}"
        
        # Guardar el archivo
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Iniciar análisis en segundo plano
        analysis_tasks[analysis_id] = {
            "status": "queued",
            "progress": 0,
            "message": "En cola para procesamiento"
        }
        
        background_tasks.add_task(procesar_cv_background, str(file_path), analysis_id, job_description)
        
        return JSONResponse({
            "message": "CV recibido y en proceso de análisis",
            "analysis_id": analysis_id
        })
        
    except Exception as e:
        logger.error(f"Error al procesar el archivo subido: {e}")
        return JSONResponse({
            "error": f"Error al procesar el archivo: {str(e)}"
        }, status_code=500)

@app.post("/api/upload-batch/")
async def upload_cv_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), job_description: str = Form(DESCRIPCION_TRABAJO)):
    """
    Endpoint para subir y analizar múltiples CVs.
    """
    try:
        if not files:
            return JSONResponse({
                "error": "No se proporcionaron archivos"
            }, status_code=400)
        
        # Generar un ID único para este lote
        batch_id = f"batch_{int(time.time())}"
        
        # Guardar los archivos
        file_paths = []
        for file in files:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))
        
        # Iniciar análisis en segundo plano
        background_tasks.add_task(procesar_cvs_batch_background, file_paths, batch_id, job_description)
        
        return JSONResponse({
            "message": f"Lote de {len(files)} CVs recibido y en proceso de análisis",
            "batch_id": batch_id
        })
        
    except Exception as e:
        logger.error(f"Error al procesar el lote de archivos: {e}")
        return JSONResponse({
            "error": f"Error al procesar los archivos: {str(e)}"
        }, status_code=500)

@app.get("/api/analysis-status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Endpoint para consultar el estado de un análisis.
    """
    if analysis_id not in analysis_tasks:
        return JSONResponse({
            "error": "ID de análisis no encontrado"
        }, status_code=404)
    
    return JSONResponse(analysis_tasks[analysis_id])

@app.get("/api/analysis-result/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    Endpoint para obtener el resultado de un análisis.
    """
    if analysis_id not in analysis_tasks:
        # Intentamos buscar el archivo de resultados
        result_file = RESULTS_DIR / f"{analysis_id}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        return JSONResponse({
            "error": "ID de análisis no encontrado"
        }, status_code=404)
    
    task_info = analysis_tasks[analysis_id]
    
    if task_info["status"] != "completed":
        return JSONResponse({
            "error": "El análisis aún no ha finalizado",
            "status": task_info["status"],
            "progress": task_info["progress"]
        }, status_code=400)
    
    return JSONResponse(task_info.get("result", {}))

@app.get("/api/batch-result/{batch_id}")
async def get_batch_result(batch_id: str):
    """A     
    Endpoint para obtener los resultados de un lote.
    """
    # Intentamos buscar el archivo de resultados
    result_file = RESULTS_DIR / f"{batch_id}.json"
    
    if not result_file.exists():
        if batch_id not in analysis_tasks:
            return JSONResponse({
                "error": "ID de lote no encontrado"
            }, status_code=404)
        
        task_info = analysis_tasks[batch_id]
        
        if task_info["status"] != "completed":
            return JSONResponse({
                "error": "El procesamiento del lote aún no ha finalizado",
                "status": task_info["status"],
                "progress": task_info["progress"]
            }, status_code=400)
    
    with open(result_file, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))

@app.get("/api/batch-summary/{batch_id}")
async def get_batch_summary(batch_id: str):
    """
    Endpoint para obtener el resumen ejecutivo de un lote.
    """
    summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
    
    if not summary_file.exists():
        if batch_id not in analysis_tasks:
            return JSONResponse({
                "error": "ID de lote no encontrado"
            }, status_code=404)
        
        task_info = analysis_tasks[batch_id]
        
        if task_info["status"] != "completed":
            return JSONResponse({
                "error": "El procesamiento del lote aún no ha finalizado",
                "status": task_info["status"],
                "progress": task_info["progress"]
            }, status_code=400)
        
        # Si está en la memoria pero no en disco
        if "summary" in task_info:
            return JSONResponse({
                "summary": task_info["summary"]
            })
    
    with open(summary_file, "r", encoding="utf-8") as f:
        return JSONResponse({
            "summary": f.read()
        })

@app.get("/view/cv/{analysis_id}", response_class=HTMLResponse)
async def view_cv_result(request: Request, analysis_id: str):
    """
    Página para visualizar el resultado del análisis de un CV.
    """
    # Buscar el resultado
    result_file = RESULTS_DIR / f"{analysis_id}.json"
    
    if not result_file.exists():
        if analysis_id not in analysis_tasks:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Análisis no encontrado"
            })
        
        task_info = analysis_tasks[analysis_id]
        
        if task_info["status"] != "completed":
            return templates.TemplateResponse("processing.html", {
                "request": request,
                "analysis_id": analysis_id,
                "status": task_info["status"],
                "progress": task_info["progress"],
                "message": task_info["message"]
            })
        
        result = task_info.get("result", {})
    else:
        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    
    return templates.TemplateResponse("cv_result.html", {
        "request": request,
        "analysis_id": analysis_id,
        "result": result
    })

@app.get("/view/batch/{batch_id}", response_class=HTMLResponse)
async def view_batch_result(request: Request, batch_id: str):
    """
    Página para visualizar los resultados de un lote de CVs.
    """
    result_file = RESULTS_DIR / f"{batch_id}.json"
    summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
    
    if not result_file.exists():
        if batch_id not in analysis_tasks:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Análisis por lotes no encontrado"
            })
        
        task_info = analysis_tasks[batch_id]
        
        if task_info["status"] != "completed":
            return templates.TemplateResponse("batch_processing.html", {
                "request": request,
                "batch_id": batch_id,
                "status": task_info["status"],
                "progress": task_info["progress"],
                "message": task_info["message"],
                "total_files": task_info.get("total_files", 0),
                "processed_files": task_info.get("processed_files", 0)
            })
        
        results = task_info.get("results", [])
        summary = task_info.get("summary", "")
    else:
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read()
    
    return templates.TemplateResponse("batch_result.html", {
        "request": request,
        "batch_id": batch_id,
        "results": results,
        "summary": summary
    })

@app.get("/job-description", response_class=HTMLResponse)
async def view_job_description(request: Request):
    """
    Página para ver y editar la descripción del trabajo.
    """
    return templates.TemplateResponse("job_description.html", {
        "request": request,
        "descripcion_trabajo": DESCRIPCION_TRABAJO
    })

@app.post("/api/update-job-description")
async def update_job_description(description: JobDescription):
    """
    Endpoint para actualizar la descripción del trabajo.
    """
    global DESCRIPCION_TRABAJO
    DESCRIPCION_TRABAJO = description.description
    
    return JSONResponse({
        "message": "Descripción del trabajo actualizada correctamente"
    })

# Ejecutar la aplicación si se ejecuta directamente
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)