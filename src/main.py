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

# Configuración del sistema de logging para registrar mensajes y errores,
# tanto en un archivo 'cv_parser.log' como en la consola estándar.
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging (INFO, DEBUG, ERROR, etc)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato del mensaje
    handlers=[
        logging.FileHandler("cv_parser.log"),  # Guardar logs en archivo
        logging.StreamHandler()  # También mostrar logs en consola
    ]
)
logger = logging.getLogger(__name__)  # Logger principal de este módulo

# Creación de la aplicación FastAPI con título y descripción visibles en la documentación automática
app = FastAPI(title="CV Parser API", description="API para análisis de currículums vitae")

# Definición de rutas base para directorios del proyecto, usando pathlib para manejo multiplataforma
BASE_DIR = Path(__file__).resolve().parent  # Carpeta base del proyecto
STATIC_DIR = BASE_DIR / "static"  # Carpeta para archivos estáticos (CSS, JS, imágenes)
TEMPLATES_DIR = BASE_DIR / "templates"  # Carpeta con plantillas HTML (Jinja2)
UPLOAD_DIR = BASE_DIR / "uploads"  # Carpeta para archivos subidos por usuarios (CVs)
RESULTS_DIR = BASE_DIR / "results"  # Carpeta para guardar resultados del análisis

# Crear las carpetas anteriores si no existen para evitar errores en escritura
for dir_path in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Configuración del motor de plantillas Jinja2 para renderizar HTML dinámico
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Montar la carpeta de archivos estáticos para servirlos desde /static en las URLs
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configuración clave API para OpenAI (se recomienda mantener esta clave segura y fuera del código)
openai.api_key = "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"

# URL base del servicio OCR.space para convertir imágenes a texto
OCR_URL = "https://api.ocr.space/parse/image"
# Clave API para el servicio OCR (similar a openai.api_key, se debe manejar con cuidado)
API_KEY = "K81778047988957"

# Texto de ejemplo con la descripción del puesto a evaluar (POC = prueba de concepto)
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
# Definición de modelos Pydantic para validar y serializar los datos usados en la API
class JobDescription(BaseModel):
    description: str = DESCRIPCION_TRABAJO  # Descripción de puesto por defecto

class CVAnalysisResult(BaseModel):
    analysis_id: str  # Identificador único del análisis
    status: str  # Estado actual (ejemplo: 'en progreso', 'completado')
    progress: int = 0  # Porcentaje de avance del análisis
    result: Optional[Dict] = None  # Resultado final del análisis (opcional)

class AnalysisStatus(BaseModel):
    status: str  # Estado general de la tarea
    progress: int  # Avance en porcentaje
    message: str  # Mensaje descriptivo del estado

class AnalysisInput(BaseModel):
    cv_id: str  # Identificador único del CV
    content: str  # Contenido resumen o análisis generado

class ChatRequest(BaseModel):
    message: str  # Mensaje que envía el usuario al chat

class ChatInput(BaseModel):
    cv_id: str  # Identificador del CV al que se asocia el chat
    message: str  # Mensaje del usuario en el chat

class ChatMessage(BaseModel):
    role: str  # Rol del mensaje: "user" o "assistant"
    content: str  # Contenido del mensaje

class ChatHistoryResponse(BaseModel):
    history: List[ChatMessage]  # Lista de mensajes del chat

# Variables globales para manejo del estado de la aplicación en memoria

analysis_tasks = {}  # Diccionario para seguimiento de análisis en background (id -> tarea)

cv_analyses = {}  # Diccionario para almacenar análisis de CVs en memoria (cv_id -> análisis)

chat_histories = {}  # Diccionario para almacenar historial de chat (cv_id -> lista de mensajes)

# Función para extraer texto de un PDF usando PyMuPDF (fitz)
def extraer_texto_pdf(ruta_archivo: str) -> str:
    try:
        # Abrimos el PDF con PyMuPDF
        doc = fitz.open(ruta_archivo)
        texto = ""
        # Recorremos cada página y extraemos el texto plano
        for pagina in doc:
            texto += pagina.get_text()
        return texto.strip()  # Eliminamos espacios en blanco al inicio y final
    except Exception as e:
        # Si hay error (archivo corrupto, no PDF, etc) lo registramos y devolvemos vacío
        logger.error(f"Error al procesar el PDF {ruta_archivo}: {e}")
        return ""

# Función para extraer texto usando OCR cuando el PDF es una imagen escaneada
def extraer_texto_ocr(ruta_archivo: str) -> str:
    try:
        with open(ruta_archivo, 'rb') as f:
            # Enviamos el archivo a la API OCR.space mediante POST
            respuesta = requests.post(
                OCR_URL,
                files={'file': f},
                data={
                    'language': 'spa',  # Idioma español para OCR
                    'isOverlayRequired': False,
                    'apikey': API_KEY
                },
                timeout=120  # Timeout elevado para archivos grandes
            )
        
        # Comprobamos código HTTP
        if respuesta.status_code != 200:
            logger.error(f"Error en la solicitud OCR. Código: {respuesta.status_code}")
            return ""
        
        # Procesamos la respuesta JSON
        resultado = respuesta.json()
        
        # Comprobamos si hubo error en OCR.space
        if resultado.get("IsErroredOnProcessing"):
            logger.error(f"Error en OCR: {resultado.get('ErrorMessage')}")
            return ""
        
        # Extraemos texto de todas las páginas analizadas y las unimos
        textos = [item["ParsedText"] for item in resultado.get("ParsedResults", [])]
        return "\n".join(textos)
        
    except Exception as e:
        logger.error(f"Error en el proceso OCR para {ruta_archivo}: {e}")
        return ""

# Función para decidir si un CV es texto o imagen y extraer el texto correspondiente
def extraer_texto(ruta_archivo: str) -> str:
    # Intentamos primero extraer texto directamente del PDF
    texto = extraer_texto_pdf(ruta_archivo)
    
    # Si el texto extraído es significativo (más de 100 caracteres), lo retornamos
    if texto and len(texto) > 100:
        logger.info(f"Texto extraído correctamente de {ruta_archivo}")
        return texto
    else:
        # Si texto corto, asumimos que el archivo es imagen o PDF escaneado y usamos OCR
        logger.info(f"El archivo {ruta_archivo} parece ser una imagen o PDF escaneado. Usando OCR.")
        return extraer_texto_ocr(ruta_archivo)

# Función para interactuar con OpenAI y extraer información estructurada de un CV
def analizar_cv_con_llm(texto: str) -> Dict:
    """
    Envía el texto del CV a OpenAI y obtiene información estructurada en JSON.
    """
    # Validamos que el texto tenga suficiente longitud para análisis válido
    if not texto or len(texto) < 50:
        logger.warning("El texto proporcionado es demasiado corto para un análisis efectivo")
        return {
            "error": "Texto insuficiente para analizar",
            "texto_original_longitud": len(texto) if texto else 0
        }
    
    try:
        # Construcción del prompt detallado para que el modelo extraiga información precisa y estructurada
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
        
        # Llamada a la API ChatCompletion con GPT-3.5 turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Modelo elegido
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en extraer información estructurada de currículums vitae."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Temperatura baja para mayor consistencia
        )
        
        # Extraemos el texto del mensaje generado por el modelo
        resultado_text = response.choices[0].message.content.strip()
        
        # Limpiamos si el resultado tiene delimitadores de bloque de código JSON
        if resultado_text.startswith("```json"):
            resultado_text = resultado_text[7:]
        if resultado_text.endswith("```"):
            resultado_text = resultado_text[:-3]
        resultado_text = resultado_text.strip()
        
        # Intentamos convertir la respuesta JSON en un diccionario Python
        try:
            resultado = json.loads(resultado_text)
            logger.info("Análisis del CV completado correctamente con LLM")
            return resultado
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de la respuesta LLM: {e}")
            logger.error(f"Texto recibido: {resultado_text[:200]}...")
            
            # Intento de limpieza adicional para caracteres no ASCII y nuevo parseo
            try:
                clean_text = re.sub(r'[^\x00-\x7F]+', '', resultado_text)
                resultado = json.loads(clean_text)
                logger.info("Análisis del CV completado tras limpieza de texto")
                return resultado
            except:
                # Si falla, devolvemos error con texto parcial para depuración
                return {"error": "Formato JSON inválido en la respuesta", "texto_parcial": resultado_text[:500]}
            
    except Exception as e:
        logger.error(f"Error al procesar con OpenAI: {str(e)}")
        return {"error": f"Error en la API de OpenAI: {str(e)}"}

# Función para evaluar la compatibilidad del CV con una descripción de trabajo dada
def evaluar_compatibilidad(cv_info: Dict, descripcion_trabajo: str = DESCRIPCION_TRABAJO) -> Dict:
    """
    Evalúa qué tan bien se ajusta un candidato a la descripción del trabajo.
    """
    try:
        # Primero se verifica si el CV contiene errores o si no tiene nombre,
        # lo que indica información insuficiente para hacer la evaluación
        if "error" in cv_info or not cv_info.get("nombre"):
            # Si falta información básica, se retorna una evaluación predeterminada
            return {
                "compatibilidad_general": 0,             # Compatibilidad 0 porque no se puede evaluar
                "recomendacion": "No evaluado",          # Mensaje indicando que no se realizó la evaluación
                "justificacion": "Información insuficiente en el CV"  # Explicación para el usuario
            }
            
        # Construcción del prompt para enviar a la API de OpenAI
        # Se pide analizar el perfil del candidato y la descripción del trabajo
        # La respuesta debe incluir puntuaciones y listas en formato JSON
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
        
        # Se realiza la llamada a la API de OpenAI con el prompt generado
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Modelo elegido para la evaluación
            messages=[
                {"role": "system", "content": "Eres un experto en recursos humanos evaluando candidatos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Baja temperatura para mayor consistencia en respuestas
        )
        
        # Se obtiene el texto de la respuesta generada por el modelo
        resultado_text = response.choices[0].message.content.strip()
        
        # Se limpia la respuesta para quitar posibles delimitadores de bloques de código JSON
        if resultado_text.startswith("```json"):
            resultado_text = resultado_text[7:]
        if resultado_text.endswith("```"):
            resultado_text = resultado_text[:-3]
        
        try:
            # Se intenta parsear la cadena JSON a un diccionario Python
            evaluacion = json.loads(resultado_text.strip())
            logger.info(f"Evaluación de compatibilidad completada para {cv_info.get('nombre', 'candidato desconocido')}")
            return evaluacion  # Retorna el resultado estructurado si todo sale bien
        except json.JSONDecodeError as e:
            # Si ocurre un error al decodificar el JSON, se registra en el log
            logger.error(f"Error al decodificar JSON de evaluación: {e}")
            # Se devuelve un resultado básico indicando que hubo problema al procesar la evaluación
            return {
                "compatibilidad_general": 50,
                "recomendacion": "No evaluado completamente",
                "justificacion": "Error al procesar la evaluación. Se requiere revisión manual."
            }
        
    except Exception as e:
        # Captura cualquier otra excepción inesperada y la registra
        logger.error(f"Error al evaluar compatibilidad: {e}")
        # Devuelve un resultado con error y compatibilidad 0 para no bloquear el proceso
        return {
            "error": f"Error en la evaluación: {str(e)}",
            "compatibilidad_general": 0,
            "recomendacion": "No evaluado"
        }

# Función para verificar la existencia de las empresas mencionadas en el CV con manejo de errores mejorado
def verificar_empresas(cv_info: Dict) -> Dict:
    """
    Verifica la existencia y reputación de las empresas mencionadas en el CV.
    """
    empresas_verificadas = []  # Lista donde se almacenarán las empresas verificadas
    
    # Si no hay experiencia previa, no hay empresas que verificar, se retorna mensaje
    if not cv_info.get("experiencia_previa"):
        return {"empresas_verificadas": [], "mensaje": "No se encontraron empresas para verificar"}
    
    # Iteramos sobre cada experiencia previa listada en el CV
    for experiencia in cv_info.get("experiencia_previa", []):
        empresa = experiencia.get("empresa", "")  # Obtenemos el nombre de la empresa
        if not empresa:
            continue  # Si no hay nombre, se omite esta experiencia
            
        logger.info(f"Verificando empresa: {empresa}")  # Log de la empresa que se está verificando
        
        try:
            # Preparar URL para consulta a la API REST de Wikipedia para obtener resumen
            empresa_normalizada = empresa.replace(' ', '_')  # Normalizar nombre para URL
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{empresa_normalizada}"
            headers = {'User-Agent': 'CVParser/1.0'}  # Header con User-Agent para la petición
            
            try:
                # Realizar la solicitud GET a Wikipedia con timeout para evitar bloqueos
                wiki_response = requests.get(wiki_url, headers=headers, timeout=10)
                # Si el status es 200 (OK), parseamos JSON, si no, dejamos dict vacío
                wiki_data = wiki_response.json() if wiki_response.status_code == 200 else {}
            except (requests.RequestException, json.JSONDecodeError) as e:
                # Capturar excepciones de red o parseo y loguear advertencia
                logger.warning(f"Error al consultar Wikipedia para {empresa}: {e}")
                wiki_data = {}  # En caso de error dejamos vacío para manejar después
            
            # Diccionario con datos básicos iniciales de la empresa no verificada
            empresa_info = {
                "nombre": empresa,
                "verificada": False,
                "fuente": "No verificada",
                "descripcion": "No se encontró información",
                "confiabilidad": "Baja"
            }
            
            # Si la respuesta de Wikipedia indica que hay una página estándar (empresa existe)
            if wiki_data.get("type") == "standard":
                # Actualizamos el dict con datos confirmados y descripción corta
                empresa_info.update({
                    "verificada": True,
                    "fuente": "Wikipedia",
                    "descripcion": wiki_data.get("extract", "")[:200] + "...",  # Extracto limitado a 200 caracteres
                    "confiabilidad": "Alta"
                })
            else:
                # Si no se encontró info estándar, marcamos como pendiente verificación manual
                empresa_info.update({
                    "verificada": False,
                    "fuente": "Búsqueda manual requerida",
                    "descripcion": "Requiere verificación manual",
                    "confiabilidad": "Pendiente"
                })
            
            # Agregamos la información de la empresa verificada o no a la lista final
            empresas_verificadas.append(empresa_info)
            
            # Pequeña pausa para no saturar la API y evitar limitaciones de tasa (throttling)
            time.sleep(0.5)
            
        except Exception as e:
            # Capturamos cualquier excepción inesperada durante la verificación
            logger.error(f"Error al verificar empresa {empresa}: {e}")
            # Añadimos la empresa a la lista con error detallado
            empresas_verificadas.append({
                "nombre": empresa,
                "verificada": False,
                "fuente": "Error",
                "descripcion": f"Error al verificar: {str(e)}",
                "confiabilidad": "Desconocida"
            })
    
    # Calculamos un score de confiabilidad general basado en porcentaje de empresas verificadas
    empresas_verificadas_count = sum(1 for e in empresas_verificadas if e["verificada"])
    total_empresas = len(empresas_verificadas)
    confiabilidad_general = (empresas_verificadas_count / total_empresas * 100) if total_empresas > 0 else 0
    
    # Retornamos un dict resumen con resultados y mensaje final
    return {
        "empresas_verificadas": empresas_verificadas,
        "total_empresas": total_empresas,
        "empresas_confirmadas": empresas_verificadas_count,
        "confiabilidad_general": confiabilidad_general,
        "mensaje": f"Se verificaron {empresas_verificadas_count} de {total_empresas} empresas"
    }

# Función para analizar un documento completo (CV), integrando extracción, análisis y evaluación
def analizar_documento(ruta_archivo: str, descripcion_trabajo: str = DESCRIPCION_TRABAJO) -> Dict:
    """
    Proceso completo de análisis de un CV: extracción de texto y análisis estructurado.
    """
    logger.info(f"Iniciando análisis del documento: {ruta_archivo}")  # Log de inicio del proceso
    
    try:
        # Extraemos el texto del documento (puede ser PDF o imagen con OCR)
        texto = extraer_texto(ruta_archivo)
        
        # Si no se pudo extraer texto, retornamos error para informar al usuario
        if not texto:
            logger.warning(f"No se pudo extraer texto de {ruta_archivo}")
            return {
                "archivo": os.path.basename(ruta_archivo),
                "error": "No se pudo extraer texto del documento"
            }
        
        # Analizamos el texto extraído con el modelo LLM para obtener datos estructurados
        resultado = analizar_cv_con_llm(texto)
        
        # Añadimos el nombre del archivo al resultado para referencia
        resultado["archivo"] = os.path.basename(ruta_archivo)
        
        # Si el análisis fue exitoso (sin errores)
        if "error" not in resultado:
            logger.info("Evaluando compatibilidad del candidato...")
            # Evaluamos la compatibilidad con la descripción de trabajo dada
            evaluacion = evaluar_compatibilidad(resultado, descripcion_trabajo)
            # Añadimos la evaluación al resultado final
            resultado["evaluacion_compatibilidad"] = evaluacion
            
            logger.info("Verificando empresas del candidato...")
            # Verificamos las empresas listadas en el CV para su existencia y reputación
            verificacion_empresas = verificar_empresas(resultado)
            # Añadimos esta verificación al resultado
            resultado["verificacion_empresas"] = verificacion_empresas
        
        # Retornamos el resultado completo con análisis, evaluación y verificación
        return resultado
        
    except Exception as e:
        # Captura cualquier error general durante el proceso
        logger.error(f"Error general en análisis de documento: {e}")
        # Retornamos error con nombre del archivo para facilitar depuración
        return {
            "archivo": os.path.basename(ruta_archivo) if ruta_archivo else "desconocido",
            "error": f"Error en el análisis: {str(e)}"
        }

# Función para generar un resumen ejecutivo basado en la lista de resultados de candidatos procesados
def generar_resumen_ejecutivo(resultados: List[Dict]) -> str:
    """
    Genera un resumen ejecutivo de todos los candidatos procesados.
    """
    # Encabezado inicial del resumen
    resumen = "=== RESUMEN EJECUTIVO DE CANDIDATOS ===\n\n"
    
    # Listas para clasificar candidatos según la recomendación o errores
    candidatos_recomendados = []
    candidatos_con_reservas = []
    candidatos_no_recomendados = []
    candidatos_con_error = []
    
    # Iteramos cada resultado para categorizarlo según su evaluación
    for resultado in resultados:
        # Si el resultado tiene un error, lo agregamos a la lista de errores
        if "error" in resultado:
            candidatos_con_error.append({
                "nombre": resultado.get("nombre", "Desconocido"),
                "archivo": resultado.get("archivo", ""),
                "error": resultado.get("error", "Error no especificado")
            })
            continue  # Pasamos al siguiente resultado sin evaluar más
        
        # Obtenemos el nombre del candidato, si no existe se pone "Desconocido"
        nombre = resultado.get("nombre", "Desconocido")
        
        # Obtenemos el diccionario de evaluación de compatibilidad (puede estar vacío)
        evaluacion = resultado.get("evaluacion_compatibilidad", {})
        
        # Extraemos la recomendación y la puntuación de compatibilidad general
        recomendacion = evaluacion.get("recomendacion", "No evaluado")
        puntuacion = evaluacion.get("compatibilidad_general", 0)
        
        # Creamos un resumen básico del candidato con los datos relevantes
        candidato_resumen = {
            "nombre": nombre,
            "puntuacion": puntuacion,
            "recomendacion": recomendacion,
            "justificacion": evaluacion.get("justificacion", "Sin evaluación")
        }
        
        # Clasificamos el candidato según la recomendación
        if recomendacion == "Altamente recomendado":
            candidatos_recomendados.append(candidato_resumen)
        elif recomendacion in ["Recomendado", "Con reservas"]:
            candidatos_con_reservas.append(candidato_resumen)
        else:
            candidatos_no_recomendados.append(candidato_resumen)
    
    # Ordenamos las listas de recomendados y con reservas por puntuación descendente
    candidatos_recomendados.sort(key=lambda x: x["puntuacion"], reverse=True)
    candidatos_con_reservas.sort(key=lambda x: x["puntuacion"], reverse=True)
    
    # Agregamos estadísticas generales al resumen
    resumen += f"Total de CVs procesados: {len(resultados)}\n"
    resumen += f"Candidatos altamente recomendados: {len(candidatos_recomendados)}\n"
    resumen += f"Candidatos con reservas: {len(candidatos_con_reservas)}\n"
    resumen += f"Candidatos no recomendados: {len(candidatos_no_recomendados)}\n"
    resumen += f"CVs con errores: {len(candidatos_con_error)}\n\n"
    
    # Detallamos candidatos altamente recomendados
    if candidatos_recomendados:
        resumen += "CANDIDATOS ALTAMENTE RECOMENDADOS:\n"
        for candidato in candidatos_recomendados:
            resumen += f"- {candidato['nombre']} (Puntuación: {candidato['puntuacion']}%)\n"
            resumen += f"  {candidato['justificacion']}\n\n"
    
    # Detallamos candidatos con reservas
    if candidatos_con_reservas:
        resumen += "CANDIDATOS CON RESERVAS:\n"
        for candidato in candidatos_con_reservas:
            resumen += f"- {candidato['nombre']} (Puntuación: {candidato['puntuacion']}%)\n"
            resumen += f"  {candidato['justificacion']}\n\n"
    
    # Detallamos CVs que tuvieron errores en el procesamiento
    if candidatos_con_error:
        resumen += "CVs CON ERRORES DE PROCESAMIENTO:\n"
        for candidato in candidatos_con_error:
            resumen += f"- Archivo: {candidato['archivo']}\n"
            resumen += f"  Error: {candidato['error']}\n\n"
    
    # Retornamos el resumen completo en formato texto
    return resumen

# Función para validar y enriquecer los resultados antes de usarlos o guardarlos
def validar_y_enriquecer_resultados(resultados: List[Dict]) -> List[Dict]:
    """
    Realiza validaciones y mejoras adicionales a los resultados obtenidos.
    """
    resultados_validados = []  # Lista donde guardaremos los resultados ya validados
    
    # Iteramos cada resultado para verificar y enriquecer datos
    for resultado in resultados:
        # Si el resultado tiene un error crítico y no tiene nombre, solo lo agregamos sin más
        if "error" in resultado and not resultado.get("nombre"):
            resultados_validados.append(resultado)
            continue
        
        # Definimos campos que siempre deben existir en el resultado
        campos_requeridos = ["nombre", "edad", "estudios", "experiencia_previa", 
                            "habilidades", "cursos_certificaciones"]
        
        # Aseguramos que cada campo requerido esté presente, si no, lo inicializamos en None
        for campo in campos_requeridos:
            if campo not in resultado:
                resultado[campo] = None
        
        # Validaciones y formateo específico para algunos campos
        
        # Si nombre existe y es string, capitalizamos la primera letra de cada palabra
        if resultado["nombre"] and isinstance(resultado["nombre"], str):
            resultado["nombre"] = ' '.join(word.capitalize() for word in resultado["nombre"].split())
        
        # Aseguramos que estos campos sean listas, si no, los convertimos a listas vacías
        if not isinstance(resultado.get("habilidades", []), list):
            resultado["habilidades"] = []
            
        if not isinstance(resultado.get("estudios", []), list):
            resultado["estudios"] = []
            
        if not isinstance(resultado.get("experiencia_previa", []), list):
            resultado["experiencia_previa"] = []
            
        if not isinstance(resultado.get("cursos_certificaciones", []), list):
            resultado["cursos_certificaciones"] = []
        
        # Añadimos un timestamp con la fecha y hora actuales del procesamiento
        resultado["fecha_procesamiento"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Agregamos el resultado validado y enriquecido a la lista final
        resultados_validados.append(resultado)
    
    # Retornamos la lista completa de resultados validados
    return resultados_validados

# Función asíncrona para procesar un CV individual en segundo plano
async def procesar_cv_background(file_path: str, analysis_id: str, job_description: str = DESCRIPCION_TRABAJO):
    """
    Procesa un CV en segundo plano y actualiza el estado del análisis.
    """
    try:
        # Indicamos que el análisis inició y estamos en la fase de extracción de texto
        analysis_tasks[analysis_id] = {
            "status": "processing",       # Estado actual
            "progress": 10,               # Progreso en porcentaje
            "message": "Extrayendo texto del CV..."  # Mensaje descriptivo
        }
        
        # Verificamos que el archivo exista, si no se lanza excepción
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        # Actualizamos progreso y mensaje antes de analizar el contenido del CV
        analysis_tasks[analysis_id]["progress"] = 30
        analysis_tasks[analysis_id]["message"] = "Analizando contenido del CV..."
        
        # Llamamos a la función que extrae y analiza el documento (PDF o imagen)
        resultado = analizar_documento(file_path, job_description)
        
        # Actualizamos el estado indicando que estamos validando resultados
        analysis_tasks[analysis_id]["progress"] = 50
        analysis_tasks[analysis_id]["message"] = "Validando resultados..."
        
        # Validamos y enriquecemos la estructura del resultado (formateo, campos faltantes, timestamp)
        resultado_validado = validar_y_enriquecer_resultados([resultado])[0]
        
        # Guardamos los resultados validados en un archivo JSON en disco
        analysis_tasks[analysis_id]["progress"] = 80
        analysis_tasks[analysis_id]["message"] = "Guardando resultados..."
        
        # Construimos la ruta donde se guardará el resultado JSON (RESULTS_DIR debe estar definido globalmente)
        result_file = RESULTS_DIR / f"{analysis_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(resultado_validado, f, indent=4, ensure_ascii=False)
        
        # Finalmente actualizamos el estado indicando que el análisis fue completado con éxito
        analysis_tasks[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Análisis completado",
            "result": resultado_validado
        }
        
        # Logueamos información de éxito
        logger.info(f"Análisis {analysis_id} completado exitosamente")
        
    except Exception as e:
        # Si ocurre algún error, lo capturamos y registramos
        logger.error(f"Error en el análisis {analysis_id}: {e}")
        
        # Actualizamos el estado indicando error y mensaje descriptivo
        analysis_tasks[analysis_id] = {
            "status": "error",
            "progress": 100,
            "message": f"Error: {str(e)}"
        }

# Función asíncrona para procesar un lote de CVs en segundo plano
async def procesar_cvs_batch_background(file_paths: List[str], batch_id: str, job_description: str = DESCRIPCION_TRABAJO):
    """
    Procesa un lote de CVs en segundo plano.
    """
    try:
        # Guardamos el total de archivos para el seguimiento de progreso
        total_files = len(file_paths)
        resultados = []  # Lista para almacenar los resultados de cada CV
        
        # Inicializamos la tarea de análisis en estado "processing" con datos iniciales
        analysis_tasks[batch_id] = {
            "status": "processing",
            "progress": 0,
            "message": f"Iniciando procesamiento de {total_files} CVs...",
            "total_files": total_files,
            "processed_files": 0
        }
        
        # Iteramos cada archivo para procesarlo uno a uno
        for i, file_path in enumerate(file_paths, 1):
            # Actualizamos el progreso proporcional al número de archivos procesados
            progress = int((i-1) / total_files * 100)
            analysis_tasks[batch_id].update({
                "progress": progress,
                "message": f"Procesando CV {i}/{total_files}: {os.path.basename(file_path)}",
                "processed_files": i-1
            })
            
            # Procesamos el documento con la función ya definida (sincrónica)
            resultado = analizar_documento(file_path, job_description)
            resultados.append(resultado)  # Agregamos resultado a la lista
            
            # Actualizamos el contador de archivos procesados
            analysis_tasks[batch_id]["processed_files"] = i
        
        # Una vez procesados todos los CVs, validamos y enriquecemos todos los resultados juntos
        resultados_validados = validar_y_enriquecer_resultados(resultados)
        
        # Generamos un resumen ejecutivo de todos los candidatos procesados
        resumen = generar_resumen_ejecutivo(resultados_validados)
        
        # Guardamos los resultados validados en un archivo JSON
        result_file = RESULTS_DIR / f"{batch_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(resultados_validados, f, indent=4, ensure_ascii=False)
        
        # Guardamos también el resumen en un archivo TXT para lectura rápida
        summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(resumen)
        
        # Actualizamos el estado final indicando que el procesamiento por lotes finalizó con éxito
        analysis_tasks[batch_id] = {
            "status": "completed",
            "progress": 100,
            "message": f"Análisis de {total_files} CVs completado",
            "total_files": total_files,
            "processed_files": total_files,
            "results": resultados_validados,
            "summary": resumen
        }
        
        # Logueamos información del éxito del procesamiento por lotes
        logger.info(f"Procesamiento por lotes {batch_id} completado exitosamente")
        
    except Exception as e:
        # Capturamos errores y registramos log
        logger.error(f"Error en el procesamiento por lotes {batch_id}: {e}")
        
        # Actualizamos estado indicando error
        analysis_tasks[batch_id] = {
            "status": "error",
            "progress": 100,
            "message": f"Error: {str(e)}"
        }

# Guardar análisis
@app.post("/analysis/")
def save_analysis(data: AnalysisInput):
    """
    Guarda el análisis generado para un CV identificado por cv_id.
    Además, inicializa el historial de chat con un mensaje del sistema
    que incluye el análisis para contexto en futuras conversaciones.
    """
    # Guardamos el contenido del análisis bajo la clave cv_id
    cv_analyses[data.cv_id] = data.content
    
    # Inicializamos el historial de chat con un mensaje de sistema que contiene el análisis
    chat_histories[data.cv_id] = [
        {"role": "system", "content": f"Eres un asistente de RRHH. Este es el análisis del CV del candidato: {data.content}"}
    ]
    
    # Retornamos confirmación de éxito
    return {"status": "análisis guardado"}

# Enviar mensaje al chat
@app.post("/chat/", response_model=ChatMessage)
def chat_with_cv_context(data: ChatInput):
    """
    Endpoint para enviar un mensaje al chat relacionado con un CV específico.
    El chat mantiene el contexto del análisis previamente guardado.
    """
    # Verificar que exista historial de chat para ese cv_id
    if data.cv_id not in chat_histories:
        # Si no existe, devolvemos error 404
        raise HTTPException(status_code=404, detail="CV no encontrado o sin análisis")

    # Agregar el mensaje del usuario al historial de chat
    chat_histories[data.cv_id].append({"role": "user", "content": data.message})

    # Llamar a la API de OpenAI para generar respuesta usando el historial completo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Se usa el modelo GPT-3.5 turbo
        messages=chat_histories[data.cv_id]  # Se pasa el historial para contexto
    )

    # Extraer la respuesta generada
    reply = response["choices"][0]["message"]

    # Guardar la respuesta en el historial para mantener contexto en futuros mensajes
    chat_histories[data.cv_id].append(reply)

    # Devolver la respuesta al cliente
    return reply

# Ver historial del chat
@app.get("/chat/{cv_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(cv_id: str):
    """
    Devuelve el historial completo de la conversación para un CV dado.
    """
    # Verificar que exista historial para ese cv_id
    if cv_id not in chat_histories:
        # Si no existe, error 404
        raise HTTPException(status_code=404, detail="CV no encontrado")

    # Retornar el historial completo
    return {"history": chat_histories[cv_id]}

# Rutas de la API FastAPI
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Página principal (HTML) de la aplicación.
    Renderiza la plantilla 'index.html'.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint para subir y analizar un CV
@app.post("/api/upload-cv/")
async def upload_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...), job_description: str = Form(DESCRIPCION_TRABAJO)):
    """
    Recibe un archivo de CV vía formulario, guarda el archivo,
    y lanza el análisis en segundo plano.
    """
    try:
        # Generar un identificador único para este análisis (con timestamp y nombre archivo)
        analysis_id = f"cv_{int(time.time())}_{file.filename.replace(' ', '_')}"

        # Guardar el archivo subido en el directorio designado
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Registrar la tarea en la estructura de control con estado 'queued'
        analysis_tasks[analysis_id] = {
            "status": "queued",
            "progress": 0,
            "message": "En cola para procesamiento"
        }

        # Agregar la tarea de análisis en segundo plano (background task)
        background_tasks.add_task(procesar_cv_background, str(file_path), analysis_id, job_description)

        # Retornar respuesta inmediata con el ID de análisis
        return JSONResponse({
            "message": "CV recibido y en proceso de análisis",
            "analysis_id": analysis_id
        })

    except Exception as e:
        # En caso de error durante la subida o procesamiento inicial
        logger.error(f"Error al procesar el archivo subido: {e}")
        return JSONResponse({
            "error": f"Error al procesar el archivo: {str(e)}"
        }, status_code=500)

@app.post("/api/upload-batch/")
async def upload_cv_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), job_description: str = Form(DESCRIPCION_TRABAJO)):
    """
    Endpoint para subir y analizar múltiples CVs en un solo lote.
    """
    try:
        # Verificamos que se hayan enviado archivos
        if not files:
            return JSONResponse({
                "error": "No se proporcionaron archivos"
            }, status_code=400)  # Error 400 si no hay archivos
        
        # Generar un ID único para este lote usando timestamp
        batch_id = f"batch_{int(time.time())}"
        
        # Guardar localmente cada archivo subido y recopilar sus rutas
        file_paths = []
        for file in files:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)  # Guardar contenido del archivo
            file_paths.append(str(file_path))  # Agregar ruta guardada a la lista
        
        # Agregar la tarea de procesamiento batch en segundo plano
        background_tasks.add_task(procesar_cvs_batch_background, file_paths, batch_id, job_description)
        
        # Retornar respuesta inmediata con info del lote y su ID
        return JSONResponse({
            "message": f"Lote de {len(files)} CVs recibido y en proceso de análisis",
            "batch_id": batch_id
        })
        
    except Exception as e:
        # En caso de error, registrar y retornar mensaje de error
        logger.error(f"Error al procesar el lote de archivos: {e}")
        return JSONResponse({
            "error": f"Error al procesar los archivos: {str(e)}"
        }, status_code=500)

@app.get("/api/analysis-status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Endpoint para consultar el estado actual de un análisis individual
    (ya sea de un solo CV o de un batch).
    """
    # Verificar si el ID existe en las tareas activas o en memoria
    if analysis_id not in analysis_tasks:
        return JSONResponse({
            "error": "ID de análisis no encontrado"
        }, status_code=404)  # No encontrado
    
    # Devolver el estado actual (status, progreso, mensaje, etc)
    return JSONResponse(analysis_tasks[analysis_id])

@app.get("/api/analysis-result/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    Endpoint para obtener el resultado final de un análisis individual.
    """
    # Si el ID no está en memoria, intentamos cargar el resultado desde archivo JSON
    if analysis_id not in analysis_tasks:
        result_file = RESULTS_DIR / f"{analysis_id}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        # Si tampoco hay archivo, respondemos error
        return JSONResponse({
            "error": "ID de análisis no encontrado"
        }, status_code=404)
    
    task_info = analysis_tasks[analysis_id]
    
    # Si el análisis aún no terminó, devolver mensaje indicando que sigue en proceso
    if task_info["status"] != "completed":
        return JSONResponse({
            "error": "El análisis aún no ha finalizado",
            "status": task_info["status"],
            "progress": task_info["progress"]
        }, status_code=400)
    
    # Si está completo, devolver el resultado guardado en memoria
    return JSONResponse(task_info.get("result", {}))

@app.get("/api/batch-result/{batch_id}")
async def get_batch_result(batch_id: str):
    """
    Endpoint para obtener los resultados de un lote de CVs (batch).
    """
    # Intentamos abrir el archivo JSON de resultados del batch
    result_file = RESULTS_DIR / f"{batch_id}.json"
    
    if not result_file.exists():
        # Si no existe archivo, verificamos si el batch está en memoria
        if batch_id not in analysis_tasks:
            # Si no está tampoco en memoria, error 404
            return JSONResponse({
                "error": "ID de lote no encontrado"
            }, status_code=404)
        
        task_info = analysis_tasks[batch_id]
        
        # Si batch está en memoria pero no ha terminado, informar progreso
        if task_info["status"] != "completed":
            return JSONResponse({
                "error": "El procesamiento del lote aún no ha finalizado",
                "status": task_info["status"],
                "progress": task_info["progress"]
            }, status_code=400)
    
    # Si archivo existe, abrir y devolver el contenido JSON con resultados
    with open(result_file, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))

@app.get("/api/batch-summary/{batch_id}")
async def get_batch_summary(batch_id: str):
    """
    Endpoint para obtener el resumen ejecutivo de un lote (batch).
    """
    summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
    
    # Si no existe el archivo resumen en disco:
    if not summary_file.exists():
        # Verificar si el batch está en memoria
        if batch_id not in analysis_tasks:
            return JSONResponse({
                "error": "ID de lote no encontrado"
            }, status_code=404)
        
        task_info = analysis_tasks[batch_id]
        
        # Si aún no finalizó el procesamiento, avisar estado
        if task_info["status"] != "completed":
            return JSONResponse({
                "error": "El procesamiento del lote aún no ha finalizado",
                "status": task_info["status"],
                "progress": task_info["progress"]
            }, status_code=400)
        
        # Si está en memoria pero no en disco, devolver el resumen guardado temporalmente
        if "summary" in task_info:
            return JSONResponse({
                "summary": task_info["summary"]
            })
    
    # Si existe archivo resumen, leer y devolverlo
    with open(summary_file, "r", encoding="utf-8") as f:
        return JSONResponse({
            "summary": f.read()
        })

@app.get("/view/cv/{analysis_id}", response_class=HTMLResponse)
async def view_cv_result(request: Request, analysis_id: str):
    """
    Página web para visualizar el resultado de análisis de un CV.
    """
    result_file = RESULTS_DIR / f"{analysis_id}.json"
    
    # Si no existe el resultado guardado en disco:
    if not result_file.exists():
        # Verificar si está en memoria
        if analysis_id not in analysis_tasks:
            # Mostrar página de error (análisis no encontrado)
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Análisis no encontrado"
            })
        
        task_info = analysis_tasks[analysis_id]
        
        # Si el análisis sigue en proceso, mostrar página de progreso
        if task_info["status"] != "completed":
            return templates.TemplateResponse("processing.html", {
                "request": request,
                "analysis_id": analysis_id,
                "status": task_info["status"],
                "progress": task_info["progress"],
                "message": task_info["message"]
            })
        
        # Si terminó pero no está en archivo, obtener resultado en memoria
        result = task_info.get("result", {})
    else:
        # Si existe archivo, cargar resultado desde JSON
        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    
    # Renderizar plantilla HTML con el resultado para mostrar al usuario
    return templates.TemplateResponse("cv_result.html", {
        "request": request,
        "analysis_id": analysis_id,
        "result": result
    })

@app.get("/view/batch/{batch_id}", response_class=HTMLResponse)
async def view_batch_result(request: Request, batch_id: str):
    """
    Página web para visualizar resultados y resumen de un lote (batch) de CVs.
    """
    result_file = RESULTS_DIR / f"{batch_id}.json"
    summary_file = RESULTS_DIR / f"{batch_id}_resumen.txt"
    
    # Si no existen los archivos en disco:
    if not result_file.exists():
        if batch_id not in analysis_tasks:
            # Mostrar error si no existe lote ni en memoria ni en disco
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Análisis por lotes no encontrado"
            })
        
        task_info = analysis_tasks[batch_id]
        
        # Si sigue en proceso, mostrar página con barra de progreso y estado
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
        
        # Si terminó y está en memoria, obtener resultados y resumen
        results = task_info.get("results", [])
        summary = task_info.get("summary", "")
    else:
        # Si archivos existen, cargar resultados y resumen desde disco
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read()
    
    # Renderizar plantilla HTML con resultados y resumen
    return templates.TemplateResponse("batch_result.html", {
        "request": request,
        "batch_id": batch_id,
        "results": results,
        "summary": summary
    })

@app.get("/job-description", response_class=HTMLResponse)
async def view_job_description(request: Request):
    """
    Página web para mostrar y permitir editar la descripción del trabajo.
    """
    return templates.TemplateResponse("job_description.html", {
        "request": request,
        "descripcion_trabajo": DESCRIPCION_TRABAJO  # Texto global editable
    })

@app.post("/api/update-job-description")
async def update_job_description(description: JobDescription):
    """
    Endpoint para actualizar la descripción del trabajo usada en análisis.
    """
    global DESCRIPCION_TRABAJO
    DESCRIPCION_TRABAJO = description.description  # Actualiza variable global
    
    return JSONResponse({
        "message": "Descripción del trabajo actualizada correctamente"
    })

# Código para ejecutar la aplicación con uvicorn si se ejecuta directamente
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
