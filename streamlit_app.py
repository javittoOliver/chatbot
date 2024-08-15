import os
import streamlit as st
import pandas as pd
from groq import Groq
import json

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
# Título de la aplicación Streamlit
st.title("Loope x- 🤖")

# Barra lateral para cargar archivo, seleccionar modelo y ajustar parámetros
with st.sidebar:
    st.write("Estás usando  **Streamlit💻** and **Groq🖥**")
    
    # Permite al usuario subir un archivo Excel
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])
    # Permite al usuario seleccionar el modelo a utilizar
    modelo = st.selectbox("Modelo", ["llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192", "gemma-7b-it"])
    # Permite al usuario ingresar un mensaje de sistema
    system_message = st.text_input("System Message", placeholder="Default : You are a helpful assistant.")
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

# Si se ha cargado un archivo, procesa y muestra su contenido
if uploaded_file is not None:
    # Lee el archivo Excel en un DataFrame de Pandas
    df = pd.read_excel(uploaded_file)
    
    # Convertir columnas que contienen objetos a cadenas para evitar problemas con Arrow
    df = df.astype({col: str for col in df.select_dtypes(include=['object']).columns})

    st.write("Contenido del archivo:")
    st.dataframe(df)

    # Convertir columnas de Timestamp a string
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    # Convierte el DataFrame a una lista de diccionarios
    lista_diccionario = df.to_dict(orient="records")

    #data_summary = f"El archivo contiene {df.shape[0]} filas y {df.shape[1]} columnas. Las columnas son: {', '.join(df.columns)}."
    #data_description = df.describe().to_string()
    
    # Convierte la lista de diccionarios a un texto JSON 
    lista_diccionario_texto = json.dumps(lista_diccionario, ensure_ascii=False, indent=2)
    
    # Permite al usuario ingresar una pregunta sobre el archivo
    prompt = st.chat_input("Haz una pregunta sobre el archivo...")

    if prompt:
        # Añade la pregunta del usuario al historial de chat
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        # Prepara el prompt para la API incluyendo los datos del archivo
        response_prompt = f"{prompt}\n\nDatos del archivo:\n{lista_diccionario_texto}"
        response = generate_content(modelo, response_prompt, system_message, max_tokens, temperature)
        
        # Muestra la respuesta generada por el asistente en streaming
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        # Añade la respuesta del asistente al historial de chat
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )
# Si no se ha cargado un archivo, permite hacer preguntas generales
if uploaded_file is None:
    prompt = st.chat_input("Haz una pregunta general...")

    if prompt:
        # Añade la pregunta del usuario al historial de chat
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt},
        )
        with st.chat_message("user"):
            st.write(prompt)
        # Genera una respuesta para la pregunta general
        response = generate_content(modelo, prompt, system_message, max_tokens, temperature)
        # Muestra la respuesta generada por el asistente en streaming
        with st.chat_message("assistant"):
            stream_generator = get_streaming_response(response)
            streamed_response = st.write_stream(stream_generator)
        # Añade la respuesta del asistente al historial de chat
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": streamed_response},
        )
