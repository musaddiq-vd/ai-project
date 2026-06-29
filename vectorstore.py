from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks, embeddings):

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


# chunks → Splitter se aaye.
# embeddings → Titan model.
# FAISS.from_documents() → Embeddings create + FAISS index create.