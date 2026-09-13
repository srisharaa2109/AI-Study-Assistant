import streamlit as st
from rag_pipeline import search_documents

st.title("AI Study Assistant")

st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    st.write("File name:", uploaded_file.name)

    question = st.text_input(
        "Ask a question about the PDF:"
    )

    if question:
        results = search_documents(question)

        st.subheader("Answer / Relevant Information")

        if results:
            for result in results:
                st.write(result)
        else:
            st.warning("No relevant information found.")