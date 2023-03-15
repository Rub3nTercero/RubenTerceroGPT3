import openai
import pandas as pd
import numpy as np
import pickle
import os
import csv

from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
from flask import Flask, redirect, render_template, request, url_for


app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETIONS_MODEL = "text-davinci-003"

df = pd.read_csv("entrevista3.csv")
df = df.set_index(["PREGUNTA"])

# PASO 1 - PREPROCESAR LOS DATOS

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

SEPARATOR = "\n* "

COMPLETIONS_API_PARAMS = {
    "temperature": 0.6,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def get_embedding(text:str, model:str) -> List[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text:str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text:str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """

    Crea una incrustación para cada fila en el marco de datos
    utilizando la API de incrustraciones de OpenAI.

    Devuelve un diccionario que mapea entre cada vector de incrustación
    y el índice de la fila a la que corresponde.

    """

    return {
        idx: get_doc_embedding(r.RESPUESTA.replace("\n", " ")) for idx, r in df.iterrows()
    }



# PASO 2 - BUSCAR INCRUSTACIONES DE DATOS SIMILARES A LA INCRUSTACIÓN DE PREGUNTAS

def vector_similarity(x: List[float], y: List[float]) -> float:
    """

    Podríamos usar la similitud del coseno o el producto
    escalar para calcular la similitud entre los vectores.
    
    En la práctica, hemos encontrado que hace poca diferencia.
    
    """

    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    
    Encuentra la incrustación de consulta para la consulta proporcionada
    y la compara con todas las incrustaciones de documentos calculados previamente
    para encontrar las secciones más relevantes.

    Devuelve la lista de secciones del documento, ordenadas por relevancia
    en orden descendente.
    
    """

    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# PASO 3 - AGREGAR SECCIONES DE DATOS RELEVANTES A LA CONSULTA

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    
    Obtener relevante
    
    """

    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Agrega contextos hasta que nos quedemos sin espacio.
        document_section = df.loc[section_index]

        chosen_sections.append(SEPARATOR + document_section.RESPUESTA.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Información de diagnóstico útil
    print(f"Secciones {len(chosen_sections)} del documento seleccionadas:")
    print("\n".join(chosen_sections_indexes))

    header = """Responda a la pregunta con la mayor sinceridad posible utilizando el contexto proporcionado y, si la respuesta o está incluida en el texto a continuación, diga "No lo sé".
    
    Contexto:

    """
    return header + "".join(chosen_sections) + "\n\n Pregunta: " + question + "\n Respuesta:"

# PASO 4 - RESPONDER LA PREGUNTA DEL USUARIO SEGÚN EL CONTEXTO

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    context_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        context_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip(" \n")

# PASO 5 - DESARROLLO WEB

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        context_embeddings = compute_doc_embeddings(df)
        animal = request.form["animal"]
        response = answer_query_with_context(animal, df, context_embeddings)
        return redirect(url_for("index", result=response))

    result = request.args.get("result")
    return render_template("index.html", result=result)

