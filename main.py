from src.load_pdf import load_pdf_text #loading the load_pdf.py
from src.chunker import split_text #loading the chunker.py
from src.embed_store import create_faiss_index #loading the embed-store.py
import numpy as np
import ollama

if __name__ == "__main__":

    # Load PDF
    pdf_path = r"D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"
    text = load_pdf_text(pdf_path)

    # Chunk text
    chunks = split_text(text)

    print(f"Total chunks created: {len(chunks)}")

    # Create embeddings + FAISS index
    index, model = create_faiss_index(chunks)

    print("FAISS index created successfully")

def retrieve_relevant_chunks(question, index, model, chunks, top_k=3):
    # Convert question to embedding
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(question_embedding, top_k)

    # Fetch matched chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]

    return retrieved_chunks


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the information below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

print("\nPDF is ready. You can now ask questions.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")

    if question.lower() in ["exit", "quit"]:
        print("Exiting chat. Bye!")
        break

    relevant_chunks = retrieve_relevant_chunks(
        question=question,
        index=index,
        model=model,
        chunks=chunks,
        top_k=3
    )

    answer = generate_answer(question, relevant_chunks)

    print("\nAnswer:\n")
    print(answer)
    print("\n" + "-" * 50 + "\n")
