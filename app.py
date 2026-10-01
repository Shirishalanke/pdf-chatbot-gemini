import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import os


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in the .env file.")
    st.stop()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="gemini_rag_collection"
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gemini RAG",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* =========================
   PAGE
   ========================= */

.stApp {
    background: #f5f7ff;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}


/* =========================
   TITLE
   ========================= */

h1 {
    color: #111827 !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    line-height: 1.2 !important;
    margin: 0 0 10px 0 !important;
}


/* Subtitle */
.block-container p {
    color: #475569;
}


/* =========================
   FILE UPLOADER
   ========================= */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #6366f1;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
}

/* Upload label */
[data-testid="stFileUploader"] label {
    color: #1e293b !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    color: #1e293b !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}


/* =========================
   NORMAL BUTTONS
   ========================= */

.stButton > button {
    background: #4f46e5 !important;
    color: white !important;

    border: none !important;
    border-radius: 10px !important;

    padding: 0.65rem 1.2rem !important;

    font-weight: 600 !important;

    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.25);

    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #4338ca !important;
    color: white !important;

    transform: translateY(-2px);

    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
}


/* =========================
   TEXT INPUT
   ========================= */

.stTextInput input {
    background: white !important;
    color: #111827 !important;

    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

.stTextInput label {
    color: #1e293b !important;
}


/* =========================
   HEADINGS
   ========================= */

h2,
h3 {
    color: #111827 !important;
}


/* =========================
   EXPANDER
   ========================= */

[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #1e293b !important;
}


/* =========================
   DIVIDER
   ========================= */

hr {
    border-top: 1px solid #dbe2ea !important;
}


/* =========================
   MOBILE
   ========================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 2.2rem !important;
    }

}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 Chat with your PDF")
st.write("RAG using Gemini API + ChromaDB")


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(text, chunk_size=1000):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end

    return chunks


# --------------------------------------------------
# GEMINI EMBEDDING
# --------------------------------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# --------------------------------------------------
# STORE CHUNKS IN CHROMADB
# --------------------------------------------------

def store_chunks(chunks):

    # Clear previous document
    try:
        chroma_client.delete_collection("gemini_rag_collection")
    except:
        pass

    global collection

    collection = chroma_client.get_or_create_collection(
        name="gemini_rag_collection"
    )

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(chunk)

        embeddings.append(embedding)

    ids = []

    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return len(chunks)


# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔄 Process PDF"):

        with st.spinner("Reading PDF..."):

            text = extract_text_from_pdf(uploaded_file)

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

        else:

            st.info(
                f"Extracted {len(text)} characters."
            )

            with st.spinner("Creating chunks..."):

                chunks = create_chunks(text)

            st.info(
                f"Created {len(chunks)} chunks."
            )

            with st.spinner(
                "Creating Gemini embeddings and storing in ChromaDB..."
            ):

                number_of_chunks = store_chunks(chunks)

            st.success(
                f"PDF processed successfully! "
                f"{number_of_chunks} chunks stored."
            )

            st.session_state.pdf_processed = True


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

if st.session_state.get("pdf_processed", False):

    st.divider()

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("🔍 Get Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching document..."):

                # Create embedding for question
                question_embedding = create_embedding(
                    question
                )

                # Search ChromaDB
                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=3
                )

            documents = results["documents"][0]

            context = "\n\n".join(documents)

            st.subheader("📖 Retrieved Context")

            with st.expander("View retrieved information"):

                for i, doc in enumerate(documents):

                    st.write(
                        f"### Chunk {i + 1}"
                    )

                    st.write(doc)

            with st.spinner("Gemini is generating the answer..."):

                answer = generate_answer(
                    question,
                    context
                )

            st.subheader("🤖 Answer")

            st.write(answer)
