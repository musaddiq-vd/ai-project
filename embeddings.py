from langchain_aws import BedrockEmbeddings


def get_embeddings():

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    )

    return embeddings


# BedrockEmbeddings → Titan Embedding model initialize karta hai.
# get_embeddings() → Embedding object return karta hai.
# Is object ko baad me FAISS use karega.