from openai import OpenAI
from streamlit import secrets
from rag_app.config import OPENAI_API_KEY
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, max_tokens=12)
embedding_func = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def load_doc(file_path: str):
    loader = PyPDFLoader(file_path)
    doc_data = loader.load()

    splitter = RecursiveCharacterTextSplitter(                          #TODO: check these later
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10
    )

    return splitter.split_text(doc_data)

def get_embeddings(texts: list[str]):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# for chunk in llm.stream("Write me a 1 verse song about sparkling water."):
#     print(chunk.text(), end="|", flush=True)