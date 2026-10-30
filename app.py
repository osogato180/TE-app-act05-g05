import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de configuración GOOGLE_API_KEY o MONGODB_URI en los secrets.")
    st.stop()

# =======================
# CLIENTES (Con caché)
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["owasp_vectors"]  # Conectado a la colección del OWASP Top 10

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES DE BÚSQUEDA Y RAG
# =======================

def crear_embedding(texto: str):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values

def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "owasp_vector_index",  # Índice vectorial de OWASP
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""Eres un asistente experto en ciberseguridad basado en el documento OWASP Top 10. Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español."""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# =======================
# INTERFAZ DE USUARIO (STREAMLIT)
# =======================

st.set_page_config(page_title="Chat OWASP Top 10 con MongoDB + Gemini", page_icon="🛡️")
st.title("🛡️ Chatbot de OWASP Top 10 (MongoDB + Gemini)")

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial de mensajes
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])

pregunta = st.chat_input("Escribe tu pregunta sobre seguridad o el OWASP Top 10...")

if pregunta:
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta en el documento OWASP..."):
            try:
                emb = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=5)

                if not similares:
                    respuesta = "No encontré información relevante en el documento OWASP Top 10."
                else:
                    respuesta = generar_respuesta(pregunta, similares)
            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error: {e}"

        st.write(respuesta)

        # Mostrar fragmentos fuente recuperados
        if 'similares' in locals() and similares:
            with st.expander("🔍 Fragmentos recuperados del documento OWASP"):
                for i, c in enumerate(similares, 1):
                    st.markdown(f"**Fragmento {i}** — Puntuación (score): `{c['score']:.4f}`")
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
