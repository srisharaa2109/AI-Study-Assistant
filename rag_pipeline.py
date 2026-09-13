import chromadb
from sentence_transformers import SentenceTransformer
from pdf_processor import extract_text_from_pdf


PDF_PATH = "data/Smart Tax Computation System Report.pdf"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="data/chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="study_documents"
)


def build_rag_database():
    pages = extract_text_from_pdf(PDF_PATH)

    documents = []
    ids = []

    for page in pages:
        text = page["text"].strip()

        if text:
            documents.append(text)
            ids.append(f"page_{page['page']}")

    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(f"Added {len(documents)} pages to ChromaDB.")


def search_documents(query, top_k=3):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]


if __name__ == "__main__":
    build_rag_database()

    print("\nDatabase created successfully!")
    print("\nTest search:")

    results = search_documents("What is the purpose of the tax computation system?")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(result[:500])