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
st.caption("Ask questions from your uploaded PDF")


# -----------------------------
# Tesseract Path
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = text.replace("\n"," ")
    text = re.sub(r"\s+"," ",text)
    return text.strip()


# -----------------------------
# Load PDF + OCR
# -----------------------------
def load_pdf_with_ocr(uploaded_file):

    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    docs = []

    for i,page in enumerate(pdf):

        native_text = page.get_text().strip()
        ocr_text = ""

        if len(native_text) < 50:

            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)

            ocr_text = pytesseract.image_to_string(img)

        full_text = clean_text(native_text + " " + ocr_text)

        if full_text:
            docs.append(
                Document(
                    page_content=full_text,
                    metadata={"page":i+1}
                )
            )

    return docs


# -----------------------------
# Build Vector DB
# -----------------------------
def build_vector_db(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db


# -----------------------------
# Extract Best Answer
# -----------------------------
def extract_best_snippet(query,retrieved_docs):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    candidate_sentences = []

    for doc in retrieved_docs:

        sentences = re.split(r'(?<=[.!?])\s+',doc.page_content)

        for sent in sentences:

            if len(sent) > 20:
                candidate_sentences.append((sent,doc.metadata))

    if not candidate_sentences:
        return "No answer found"

    query_embedding = model.encode(query,convert_to_tensor=True)

    texts = [s[0] for s in candidate_sentences]

    sent_embeddings = model.encode(texts,convert_to_tensor=True)

    scores = util.cos_sim(query_embedding,sent_embeddings)[0]

    best_idx = int(scores.argmax())

    best_sentence,meta = candidate_sentences[best_idx]

    return {
        "answer":best_sentence,
        "page":meta.get("page","unknown")
    }


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


if uploaded_file:

    if "db" not in st.session_state:

        with st.spinner("Processing PDF..."):

            docs = load_pdf_with_ocr(uploaded_file)
            db = build_vector_db(docs)

            st.session_state.db = db

        st.success("PDF loaded successfully")


# -----------------------------
# Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Show Chat History
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
query = st.chat_input("Ask something from the PDF...")


if query:

    st.chat_message("user").markdown(query)

    st.session_state.messages.append({
        "role":"user",
        "content":query
    })


    db = st.session_state.db

    retriever = db.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(query)

    result = extract_best_snippet(query,docs)

    answer = f"{result['answer']} (Page {result['page']})"


    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })