import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("📚 AI Study Assistant")
st.write("Upload a PDF and ask questions about its content.")


# -----------------------------
# Load AI model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success("✅ PDF uploaded successfully!")
    st.write("**File name:**", uploaded_file.name)

    # -----------------------------
    # Read PDF
    # -----------------------------
    pdf_bytes = uploaded_file.read()

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    # Extract text from all pages
    text = ""

    for page in pdf:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text + "\n"

    pdf.close()


    # -----------------------------
    # Check extracted text
    # -----------------------------
    if not text.strip():

        st.error(
            "❌ No readable text was found in this PDF."
        )

    else:

        # -----------------------------
        # Split text into paragraphs
        # -----------------------------
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]


        # -----------------------------
        # Create larger chunks
        # -----------------------------
        chunks = []

        current_chunk = ""

        for paragraph in paragraphs:

            # Keep paragraphs together
            if len(current_chunk) + len(paragraph) <= 3000:

                current_chunk += paragraph + "\n"

            else:

                if current_chunk.strip():
                    chunks.append(
                        current_chunk.strip()
                    )

                current_chunk = paragraph + "\n"


        # Add the final chunk
        if current_chunk.strip():

            chunks.append(
                current_chunk.strip()
            )


        # -----------------------------
        # Create ChromaDB
        # -----------------------------
        client = chromadb.Client()

        collection = client.get_or_create_collection(
            name="uploaded_pdf"
        )


        # -----------------------------
        # Create embeddings
        # -----------------------------
        with st.spinner(
            "🧠 Processing the PDF..."
        ):

            embeddings = model.encode(
                chunks
            ).tolist()


        # -----------------------------
        # Store chunks
        # -----------------------------
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )


        st.success(
            f"✅ PDF processed successfully! "
            f"{len(chunks)} sections created."
        )


        # -----------------------------
        # Ask question
        # -----------------------------
        question = st.text_input(
            "🔎 Ask a question about the PDF:"
        )


        if question.strip():

            with st.spinner(
                "🔍 Searching the PDF..."
            ):

                # Create question embedding
                question_embedding = model.encode(
                    [question]
                ).tolist()[0]


                # Search ChromaDB
                results = collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=min(
                        8,
                        len(chunks)
                    )
                )


            # -----------------------------
            # Display result
            # -----------------------------
            st.subheader(
                "📖 Relevant Information from the PDF"
            )


            documents = results.get(
                "documents",
                [[]]
            )[0]


            if documents:

                for i, result in enumerate(
                    documents,
                    start=1
                ):

                    st.markdown(
                        f"### Section {i}"
                    )

                    st.write(result)

                    st.divider()

            else:

                st.warning(
                    "No relevant information found."
                )
