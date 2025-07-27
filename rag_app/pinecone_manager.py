import time
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_REGION, PINECONE_INDEX
from config import namespace


pc = Pinecone(api_key=PINECONE_API_KEY)
indexes_list = [pc_index['name'] for pc_index in pc.list_indexes()]

if PINECONE_INDEX in indexes_list:
    index = pc.Index(PINECONE_INDEX)


def pc_create_index_if_not_exists():
    if PINECONE_INDEX in indexes_list:
        return

    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_REGION,
        )
    )
    time.sleep(5)  # Wait for the index to be created


def upsert_to_pinecone(vectors: list[tuple[str, list[float], dict]]):
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )


def query_index(vector: list[float]):
    try:
        query_results = index.query(
            top_k=8,
            namespace=namespace,
            vector=vector,
            include_metadata=True
        )
        return query_results

    except Exception as e:
        raise f"Error querying Pinecone index: {e}"
