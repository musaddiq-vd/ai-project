from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load PDF and return list of Document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents

# Short Explanation
# PyPDFLoader() → PDF read karega.
# loader.load() → List of Document objects return karega.
# return documents → Ye list next module (splitter) ko milegi.