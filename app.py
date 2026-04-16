import streamlit as st
import fitz
import pytesseract
from PIL import Image
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot")
st.caption("Upload a PDF and ask questions from its content")


# -----------------------------
# Tesseract Path
# -----------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# -----------------------------
# Cached Resources
# -----------------------------
@st.cache_resource
def get_sentence_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


@st.cache_resource
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


# -----------------------------
# Helpers
# -----------------------------
def clean_text(text: str) -> str:
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_text_from_page(page: fitz.Page) -> str:
    text = page.get_text().strip()
    if len(text) > 50:
        return text

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image).strip()


def load_pdf_with_ocr(pdf_bytes: bytes) -> list[Document]:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    documents: list[Document] = []

    for page_index, page in enumerate(pdf):
        raw_text = extract_text_from_page(page)
        clean_page_text = clean_text(raw_text)

        if clean_page_text:
            documents.append(
                Document(
                    page_content=clean_page_text,
                    metadata={"page": page_index + 1}
                )
            )

    return documents


def build_vector_db(documents: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def extract_best_snippet(query: str, retrieved_docs: list[Document]) -> dict[str, str]:
    sentence_model = get_sentence_model()
    candidate_sentences: list[tuple[str, dict]] = []

    for doc in retrieved_docs:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content):
            sentence = sentence.strip()
            if len(sentence) > 30:
                candidate_sentences.append((sentence, doc.metadata))

    if not candidate_sentences:
        return {"answer": "No answer found.", "page": "unknown"}

    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    texts = [sentence for sentence, _ in candidate_sentences]
    text_embeddings = sentence_model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, text_embeddings)[0]
    best_index = int(scores.argmax())
    best_sentence, meta = candidate_sentences[best_index]

    return {
        "answer": best_sentence,
        "page": str(meta.get("page", "unknown"))
    }


# -----------------------------
# Session State Initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "db" not in st.session_state:
    st.session_state.db = None


# -----------------------------
# PDF Upload and Indexing
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.file_name = uploaded_file.name

if st.session_state.pdf_bytes is not None and st.session_state.db is None:
    with st.spinner("Indexing PDF content, this may take a moment..."):
        docs = load_pdf_with_ocr(st.session_state.pdf_bytes)
        if not docs:
            st.error("No readable text was extracted from the PDF. Please upload a different file.")
        else:
            st.session_state.db = build_vector_db(docs)
            st.session_state.page_count = len(docs)
            st.success("PDF loaded and indexed successfully.")

if st.session_state.db is None:
    st.info("Upload a PDF to start asking questions.")


# -----------------------------
# Chat Interface
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask something from the PDF...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    if st.session_state.db is None:
        response = "Please upload and index a PDF before asking questions."
    else:
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)
        result = extract_best_snippet(query, retrieved_docs)
        response = f"{result['answer']}  \n(Page {result['page']})"
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
