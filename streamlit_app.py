import os
import numpy as np
import streamlit as st
import pandas as pd
from groq import Groq
import torch
import whisper
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import json
import io
import soundfile as sf
import matplotlib.pyplot as plt
import datetime

# Configura la página de Streamlit para que use todo el ancho disponible
st.set_page_config(layout="wide")


# Establece la clave API para acceder a la API de Groq desde st.secrets
api_key = st.secrets["general"]["GROQ_API_KEY"]

# Inicializa el cliente de Groq usando la clave API
client = Groq(
    api_key=api_key,
)

def get_streaming_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Función para generar contenido a partir de un modelo Groq
def generate_content(modelo:str, prompt:str, system_message:str="You are a helpful assistant.", max_tokens:int=1024, temperature:int=0.5):
    # Incluye el historial de chat en los mensajes
    messages = [{"role": "system", "content": system_message}]
    messages += st.session_state["chat_history"]
    messages.append({"role": "user", "content": prompt})
    
    stream = client.chat.completions.create(
        messages=messages,
        model=modelo,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stop=None,
        stream=True
    ) 
    return stream

# Función para transcribir audio usando Whisper

def transcribir_audio_por_segmentos(uploaded_audio):
    
        # Verificar si la GPU admite FP16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
        fp16_available = True
    else:
         fp16_available = False
    
    model = whisper.load_model("small")

    if fp16_available:
        result = model.transcribe(audio_path)
    else:
        result = model.transcribe(audio_path, fp16=False)   
    
    #model = whisper.load_model("base")
    print("Whisper model loaded.")
    result = model.transcribe(uploaded_audio)
    result = pd.DataFrame(transcripcion['segments'])[['start', 'end','text']]
    
    # Función para convertir segundos a hh:mm:ss
    def seconds_to_time(seconds):
        return str(datetime.timedelta(seconds=seconds))

    # Aplicar la función a las columnas 'start' y 'end'
    result['start'] = result['start'].apply(seconds_to_time)
    result['end'] = result['end'].apply(seconds_to_time)
    return result

# Título de la aplicación Streamlit
st.title("Vitto x- 🤖")

# Barra lateral para cargar archivo, seleccionar modelo y ajustar parámetros
with st.sidebar:
    st.write("Estás usando  **Streamlit💻** and **Groq🖥**\n from Vitto ✳️")
    
    # Permite al usuario subir un archivo Excel
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])

    # Permite al usuario subir un archivo de audio
    uploaded_audio = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "ogg", "flac"])

    # Permite al usuario seleccionar el modelo a utilizar
    modelo = st.selectbox("Modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"])

    # Permite al usuario ingresar un mensaje de sistema
    system_message = st.text_input("System Message", placeholder="Default : Eres un asistente amigable.")
    
    # Ajusta la temperatura del modelo para controlar la creatividad
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.2)
    
    # Selecciona el número máximo de tokens para la respuesta
    max_tokens = st.selectbox("Max New Tokens", [1024, 2048, 4096, 8196])

# Inicializa el historial de chat en el estado de sesión si no existe
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Muestra los mensajes del historial de chat
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Inicializa el estado de sesión si no existe
if "transcripcion_finalizada" not in st.session_state:
    st.session_state["transcripcion_finalizada"] = False
if "transcripcion" not in st.session_state:
    st.session_state["transcripcion"] = ""

# Si se ha cargado un archivo de audio, lo transcribe y muestra un mensaje cuando ha terminado
if uploaded_audio is not None and not st.session_state["transcripcion_finalizada"]:
    st.write("Transcribiendo el audio...")
    
    
    # Intenta transcribir el audio
    transcripcion = transcribir_audio_por_segmentos(uploaded_audio, segment_duration=30)
        
    # Muestra un mensaje de que la transcripción ha finalizado
    st.write("La transcripción ha finalizado. Puedes hacer preguntas sobre el contenido.")
        
    # Guardar la transcripción en el estado de sesión para referencia futura
    st.session_state["transcripcion"] = transcripcion
        
    # Marcar en el estado de sesión que la transcripción ha terminado
    st.session_state["transcripcion_finalizada"] = True

   

# Mostrar la caja de texto para hacer preguntas solo si la transcripción ha finalizado
if st.session_state["transcripcion_finalizada"] and uploaded_audio is not None:
    prompt = st.chat_input("Haz una pregunta sobre la transcripción...")

    if prompt:
        # Añade la pregunta del usuario al historial de chat
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepara el prompt para el modelo
        if st.session_state["transcripcion"]:
            response_prompt = f"{prompt}\n\nTexto transcrito:\n{st.session_state['transcripcion']}"
        else:
            response_prompt = prompt
        
        try:
            # Genera la respuesta para la pregunta del usuario
            response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)
            
            # Muestra la respuesta generada por el asistente en streaming
            with st.chat_message("assistant"):
                stream_generator = get_streaming_response(response)
                streamed_response = st.write_stream(stream_generator)
            
            # Añade la respuesta del asistente al historial de chat
            st.session_state["chat_history"].append({"role": "assistant", "content": streamed_response})
        
        except Exception as e:
            st.error("Ocurrió un error al generar la respuesta. Por favor, intenta nuevamente.")

# Si se ha cargado un archivo Excel, procesa y muestra su contenido
if uploaded_file is not None:
    try:
        # Carga el archivo Excel en un DataFrame
        dfs = pd.read_excel(uploaded_file)
        
        # Convertir columnas de texto a tipo str
        df = dfs.astype({col: str for col in dfs.select_dtypes(include=['object']).columns})
        
        # Muestra el contenido del archivo en la interfaz
        st.write("Contenido parcial del archivo:")
        st.dataframe(df.head(6))

        # Convierte columnas de datetime a str si existen
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].astype(str)
        
        # Convertir DataFrame a lista de diccionarios
        lista_diccionario = df.to_dict(orient="records")
        lista_diccionario_texto = json.dumps(lista_diccionario, ensure_ascii=False, indent=2)

        # Inicializa el modelo para interactuar con PandasAI
        llm = ChatGroq(model_name=modelo, api_key=api_key)
        smart_df = SmartDataframe(dfs, config={'llm': llm})

        # Solicita preguntas separadas para cada barra de chat
        prompt_pandasai = st.chat_input("Haz una petición para el archivo (PandasAI)...")
        prompt_dict = st.chat_input("Haz una pregunta sobre el archivo (Diccionario)...")
        

        if prompt_pandasai:
            # Agrega la consulta actual al historial de chat
            st.session_state["chat_history"].append({"role": "user", "content": prompt_pandasai})
                                                  
            with st.chat_message("user"):
                st.write(prompt_pandasai)
        
            # Construye el prompt con el historial como contexto
            combined_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["chat_history"][:-1]])
            current_question = f"{st.session_state['chat_history'][-1]['role']}: {st.session_state['chat_history'][-1]['content']}"
        
            # Prompt final con una instrucción clara sobre el idioma
            code_prompt = (
                f"Considera la siguiente conversación previa como contexto y responde solo a la consulta actual. "
                f"Responde en español.\n\n"
                f"Contexto:\n{combined_history}\n\n"
                f"Consulta actual:\n{current_question}"
            )
            
            response_pandasai = smart_df.chat(code_prompt)
        
            with st.chat_message("assistant"):
                st.write(response_pandasai)
        
            # Agrega la respuesta al historial de chat
            st.session_state["chat_history"].append({"role": "assistant", "content": response_pandasai})
            
            # Verificar si el archivo existe
            if os.path.exists("exports/charts/temp_chart.png"):
                st.image("exports/charts/temp_chart.png")
                os.remove("exports/charts/temp_chart.png")
            else:
                if 'response' in locals():
                    st.write(response)
                else:
                    st.write("")  

        if prompt_dict:
            st.session_state["chat_history"].append({"role": "user", "content": prompt_dict})
            with st.chat_message("user"):
                st.write(prompt_dict)

            response_prompt = f"{prompt_dict}\n\nDatos del archivo:\n{lista_diccionario_texto}"
            response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)

            with st.chat_message("assistant"):
                stream_generator = get_streaming_response(response)
                streamed_response = st.write_stream(stream_generator)

            st.session_state["chat_history"].append({"role": "assistant", "content": streamed_response})

    except Exception as e:
        # Muestra un mensaje de error simple en caso de que ocurra un problema
        st.error("Ocurrió un error al procesar el archivo. Por favor, intenta de nuevo.")

# Si no se ha cargado un archivo, permite hacer preguntas generales
if uploaded_file is None and uploaded_audio is None:
    prompt = st.chat_input("Haz una pregunta general...")

    if prompt:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        
        response = generate_content(modelo, prompt, system_message, max_tokens, temperature)
        
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )
