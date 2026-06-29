from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def create_rag_chain(
    llm,
    retriever,
):

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.

Context:
{context}

Question:
{input}
""")

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    return rag_chain