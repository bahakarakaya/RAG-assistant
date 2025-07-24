from openai import OpenAI
import tiktoken
from streamlit import secrets
from config import OPENAI_API_KEY
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


client = OpenAI(api_key=OPENAI_API_KEY)
#llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini",max_tokens=12)
embedding_func = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def load_doc(file_path: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    doc_data = loader.load()

    doc_data_str = ""
    for page in doc_data:
         doc_data_str += page.page_content

    splitter = RecursiveCharacterTextSplitter(                          #TODO: check these later
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10
    )

    return splitter.split_text(doc_data_str)

def get_embeddings(text: list[str]) -> list[float]:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    tokens = []
    for chunk in text:
        tokens += encoding.encode(chunk)

    response = client.embeddings.create(
        input=tokens,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# for chunk in llm.stream("Write me a 1 verse song about sparkling water."):
#     print(chunk.text(), end="|", flush=True)