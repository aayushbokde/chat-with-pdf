from typing import List
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

import google.generativeai as genai

# Load API key from environment
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')



# ========== Text Chunking ==========
def split_text_into_chunks(text, max_chunk_length=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks 


# ========== PDF Text Extraction ==========
def extract_text_from_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype='pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ========== Search for Relevant Chunks ==========
def search_index(query: str, vectorizer, index, texts: List[str], top_k: int = 5) -> List[str]:
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, index).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_indices]


# ========== Local Lightweight QA (optional fallback) ==========
qa_model = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

def generate_local_answer(question, context_chunks):
    context = " ".join(context_chunks)
    result = qa_model(question=question, context=context)
    return result["answer"]


# ========== Gemini Answer Generator ==========
def generate_gemini_answer(question, context_chunks):
    try:
        context = " ".join(context_chunks)
        context = context[:4000]
        prompt = f"""Answer the following question using the given context.

Context:
{context}

Question:
{question}

Answer:"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error while generating answer: {e}"


# ========== Gemini Summarizer ==========
def summarize_text_with_gemini(text):
    prompt = f"""Please summarize the following content briefly:

{text[:4000]}

Summary:"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
