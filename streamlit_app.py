import os
import streamlit as st
import pandas as pd
from groq import Groq
import json
import io
import numpy as np
import soundfile as sf
import openai


# Configura la página de Streamlit para que use todo el ancho disponible
st.set_page_config(layout="wide")

# Establece la clave API para acceder a la API de Groq
os.environ['GROQ_API_KEY'] = "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"

# Inicializa el cliente de Groq usando la clave API
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Función para obtener respuestas en streaming desde la API
def get_streaming_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Función para generar contenido a partir de un modelo Groq
def generate_content(modelo:str, prompt:str, system_message:str="You are a helpful assistant.", max_tokens:int=1024, temperature:int=0.5):
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        model=modelo,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stop=None,
        stream=True
    ) 
    return stream

# Función para transcribir audio usando Whisper

def transcribir_audio(uploaded_audio):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

# Título de la aplicación Streamlit
st.title("Loope x- 🤖")

# Barra lateral para cargar archivo, seleccionar modelo y ajustar parámetros
with st.sidebar:
    st.write("Estás usando  **Streamlit💻** and **Groq🖥**\n from Vitto ✳️")
    
    # Permite al usuario subir un archivo Excel
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])

    # Permite al usuario subir un archivo de audio
    uploaded_audio = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "ogg", "flac"])

    # Permite al usuario seleccionar el modelo a utilizar
    modelo = st.selectbox("Modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])

    # Permite al usuario ingresar un mensaje de sistema
    system_message = st.text_input("System Message", placeholder="Default : Eres una asistente amigable.")
    
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
    
    # Transcribe el audio
    transcripcion = transcribir_audio(uploaded_audio)
    
    # Muestra un mensaje de que la transcripción ha finalizado
    st.write("La transcripción ha finalizado. Puedes hacer preguntas sobre el contenido.")

    # Guardar la transcripción en el estado de sesión para referencia futura
    st.session_state["transcripcion"] = transcripcion

    # Marcar en el estado de sesión que la transcripción ha terminado
    st.session_state["transcripcion_finalizada"] = True

# Mostrar la caja de texto para hacer preguntas solo si la transcripción ha finalizado
if st.session_state["transcripcion_finalizada"] and  uploaded_audio is not None:
    prompt = st.chat_input("Haz una pregunta sobre la transcripción...")

    if prompt:
        # Añade la pregunta del usuario al historial de chat
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepara el prompt para el modelo
        if st.session_state["transcripcion"]:
            response_prompt = f"{prompt}\n\nTexto transcrito:\n{st.session_state['transcripcion']}"
        else:
            response_prompt = prompt
        
        # Genera la respuesta para la pregunta del usuario
        response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)
        
        # Muestra la respuesta generada por el asistente en streaming
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        
        # Añade la respuesta del asistente al historial de chat
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )

# Si se ha cargado un archivo Excel, procesa y muestra su contenido
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df = df.astype({col: str for col in df.select_dtypes(include=['object']).columns})
    st.write("Contenido del archivo:")
    st.dataframe(df)

    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    lista_diccionario = df.to_dict(orient="records")
    lista_diccionario_texto = json.dumps(lista_diccionario, ensure_ascii=False, indent=2)
    
    prompt = st.chat_input("Haz una pregunta sobre el archivo...")

    if prompt:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        
        response_prompt = f"{prompt}\n\nDatos del archivo:\n{lista_diccionario_texto}"
        response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)
        
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )

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
