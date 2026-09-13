import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Study Assistant")

st.title("AI Study Assistant")
st.write("Upload a PDF and ask questions about it.")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success("PDF uploaded successfully!")
    st.write("File name:", uploaded_file.name)

    # Read PDF
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Extract text
    text = ""

    for page in pdf:
        text += page.get_text() + "\n"

    # Split text into chunks
    chunk_size = 2000
    overlap = [
    chunks = []
    start = 0
    while start<len(text):
        end = start + chunk_size
        chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk)

    start += chunk_size - overlap
   
    ]

    if chunks:

        # Create ChromaDB collection
        client = chromadb.Client()

        collection = client.get_or_create_collection(
            name="uploaded_pdf"
        )

        # Create embeddings
        embeddings = model.encode(chunks).tolist()

        # Add document chunks
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        question = st.text_input(
            "Ask a question about the PDF:"
        )

        if question:

            # Convert question into embedding
            question_embedding = model.encode(
                [question]
            ).tolist()[0]

            # Search relevant chunks
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=min(5, len(chunks))
            )

            st.subheader("Answer / Relevant Information")

            documents = results.get("documents", [[]])[0]

            if documents:
                for result in documents:
                    st.write(result)
                    st.divider()
            else:
                st.write("No relevant information found.")

    else:
        st.warning("Could not extract text from this PDF.")
