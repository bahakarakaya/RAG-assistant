from operator import index

from pinecone import Pinecone, ServerlessSpec
from rag_app.config import PINECONE_API_KEY, PINECONE_REGION, PINECONE_INDEX
from interface import namespace

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX in pc.list_indexes():
    index = pc.Index(PINECONE_INDEX)

def pc_create_index_if_not_exists():
    if PINECONE_INDEX in pc.list_indexes():
        pass
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

def upsert_to_pinecone(vectors: list):
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )

def query_index(vector: list[float]):
    query_results = index.query(
        top_k=10,                                   #TODO: examine this later
        namespace=namespace,
        vector=vector
    )