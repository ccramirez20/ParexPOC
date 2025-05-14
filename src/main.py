import os
import json
import requests
import openai
import fitz  # PyMuPDF para trabajar con PDFs
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import re
import time

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

# Configuración de la API de Azure OpenAI
openai.api_key = "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"  # Reemplaza con tu clave real

# Definimos las rutas: carpeta con los CVs y archivo de salida
CARPETA_CVS = "./data"
ARCHIVO_SALIDA = "informacion_cvs.json"

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

# Función para interactuar con Azure OpenAI y extraer información estructurada
def analizar_cv_con_llm(texto: str) -> Dict:
    """
    Envía el texto del CV a Azure OpenAI y obtiene información estructurada.
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
            model="gpt-3.5-turbo",  # O el despliegue que estés utilizando
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
            return {"error": "Formato JSON inválido en la respuesta", "texto_parcial": resultado_text[:500]}
            
    except Exception as e:
        logger.error(f"Error al procesar con Azure OpenAI: {str(e)}")
        return {"error": f"Error en la API de Azure OpenAI: {str(e)}"}

# NUEVO: Función para evaluar la compatibilidad con la descripción de trabajo
def evaluar_compatibilidad(cv_info: Dict, descripcion_trabajo: str = DESCRIPCION_TRABAJO) -> Dict:
    """
    Evalúa qué tan bien se ajusta un candidato a la descripción del trabajo.
    """
    try:
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
        
        evaluacion = json.loads(resultado_text.strip())
        logger.info(f"Evaluación de compatibilidad completada para {cv_info.get('nombre', 'candidato desconocido')}")
        return evaluacion
        
    except Exception as e:
        logger.error(f"Error al evaluar compatibilidad: {e}")
        return {
            "error": f"Error en la evaluación: {str(e)}",
            "compatibilidad_general": 0,
            "recomendacion": "No evaluado"
        }

# NUEVO: Función para verificar la existencia de las empresas
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
            # Usando la API de Clearbit (versión gratuita) para obtener información de empresas
            # Nota: Esta API tiene límites en su versión gratuita
            # Como alternativa, también se puede usar Google Places API
            
            # Intentamos primero con una búsqueda simple en Wikipedia API (completamente gratis)
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{empresa.replace(' ', '_')}"
            headers = {'User-Agent': 'CVParser/1.0'}
            
            wiki_response = requests.get(wiki_url, headers=headers, timeout=10)
            
            empresa_info = {
                "nombre": empresa,
                "verificada": False,
                "fuente": "No verificada",
                "descripcion": "No se encontró información",
                "confiabilidad": "Baja"
            }
            
            if wiki_response.status_code == 200:
                wiki_data = wiki_response.json()
                if wiki_data.get("type") == "standard":
                    empresa_info.update({
                        "verificada": True,
                        "fuente": "Wikipedia",
                        "descripcion": wiki_data.get("extract", "")[:200] + "...",
                        "confiabilidad": "Alta"
                    })
            else:
                # Como respaldo, intentamos con Google Search (usando SerpAPI versión gratuita)
                # Nota: SerpAPI requiere API key, aquí usamos un enfoque simple
                google_url = f"https://www.google.com/search?q={requests.utils.quote(empresa + ' company')}"
                
                # Para un POC, simplemente marcamos como "verificación pendiente"
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

# Función para analizar un documento completo
def analizar_documento(ruta_archivo: str) -> Dict:
    """
    Proceso completo de análisis de un CV: extracción de texto y análisis estructurado.
    """
    logger.info(f"Iniciando análisis del documento: {ruta_archivo}")
    
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
    
    # Evaluamos la compatibilidad con el trabajo
    if "error" not in resultado:
        logger.info("Evaluando compatibilidad del candidato...")
        evaluacion = evaluar_compatibilidad(resultado)
        resultado["evaluacion_compatibilidad"] = evaluacion
        
        logger.info("Verificando empresas del candidato...")
        verificacion_empresas = verificar_empresas(resultado)
        resultado["verificacion_empresas"] = verificacion_empresas
    
    return resultado

# Procesa todos los CVs en la carpeta especificada
def procesar_todos_los_cvs(carpeta: str) -> List[Dict]:
    """
    Procesa todos los documentos PDF en la carpeta especificada.
    """
    resultados = []
    archivos_pdf = [f for f in os.listdir(carpeta) if f.lower().endswith(".pdf")]
    
    if not archivos_pdf:
        logger.warning(f"No se encontraron archivos PDF en {carpeta}")
        return []
    
    logger.info(f"Encontrados {len(archivos_pdf)} archivos PDF para procesar")
    
    for i, archivo in enumerate(archivos_pdf, 1):
        logger.info(f"Procesando archivo {i}/{len(archivos_pdf)}: {archivo}")
        ruta = os.path.join(carpeta, archivo)
        resultado = analizar_documento(ruta)
        resultados.append(resultado)
        logger.info(f"Completado análisis de {archivo}")
    
    return resultados

# Guarda el resultado en un archivo JSON estructurado
def guardar_json(data: List[Dict], archivo_salida: str):
    """
    Guarda los resultados en un archivo JSON con formato legible.
    """
    try:
        with open(archivo_salida, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Resultados guardados correctamente en {archivo_salida}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar el archivo JSON: {e}")
        return False

# Función para generar un resumen ejecutivo
def generar_resumen_ejecutivo(resultados: List[Dict]) -> str:
    """
    Genera un resumen ejecutivo de todos los candidatos procesados.
    """
    resumen = "=== RESUMEN EJECUTIVO DE CANDIDATOS ===\n\n"
    candidatos_recomendados = []
    candidatos_con_reservas = []
    candidatos_no_recomendados = []
    
    for resultado in resultados:
        if "error" in resultado:
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
    resumen += f"Candidatos no recomendados: {len(candidatos_no_recomendados)}\n\n"
    
    if candidatos_recomendados:
        resumen += "CANDIDATOS ALTAMENTE RECOMENDADOS:\n"
        for candidato in candidatos_recomendados:
            resumen += f"- {candidato['nombre']} (Puntuación: {candidato['puntuacion']}%)\n"
            resumen += f"  {candidato['justificacion']}\n\n"
    
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

# Punto de entrada principal del script
def main():
    """
    Función principal que orquesta todo el proceso.
    """
    logger.info("======= INICIANDO ANÁLISIS DE CVs =======")
    logger.info(f"Carpeta de entrada: {CARPETA_CVS}")
    logger.info(f"Archivo de salida: {ARCHIVO_SALIDA}")
    
    # Verificamos que la carpeta exista
    if not os.path.exists(CARPETA_CVS):
        logger.error(f"La carpeta {CARPETA_CVS} no existe")
        return
    
    # Procesamos todos los CVs
    resultados = procesar_todos_los_cvs(CARPETA_CVS)
    
    # Validamos y enriquecemos los resultados
    resultados_finales = validar_y_enriquecer_resultados(resultados)
    
    # Guardamos los resultados
    exito = guardar_json(resultados_finales, ARCHIVO_SALIDA)
    
    # Generamos un resumen ejecutivo
    resumen = generar_resumen_ejecutivo(resultados_finales)
    
    # Guardamos el resumen en un archivo separado
    with open("resumen_ejecutivo.txt", "w", encoding="utf-8") as f:
        f.write(resumen)
    
    if exito:
        logger.info(f"Proceso completado exitosamente. Se procesaron {len(resultados_finales)} documentos.")
        logger.info("Resumen ejecutivo guardado en 'resumen_ejecutivo.txt'")
        print("\n" + resumen)
    else:
        logger.error("Error al guardar los resultados.")
    
    logger.info("======= FIN DEL PROCESO =======")

if __name__ == "__main__":
    main()
