from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from rag_app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_REGION, PINECONE_INDEX

pc = Pinecone(api_key=PINECONE_API_KEY)

def pc_create_index_if_not_exists():
    if PINECONE_INDEX in pc.list_indexes():
        return
    else:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION,
            )
        )
