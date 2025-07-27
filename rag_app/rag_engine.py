from openai import OpenAI
import tiktoken
from config import OPENAI_API_KEY
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from pinecone_manager import query_index

client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0.0, max_tokens=512)
embedding_func = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")


def load_doc(file_path: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    doc_data = loader.load()

    doc_data_str = ""
    for page in doc_data:
        doc_data_str += page.page_content

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=15
    )

    return splitter.split_text(doc_data_str)


def get_embeddings(text: list[str] | str) -> list[float]:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    if isinstance(text, list):
        tokens = []
        for chunk in text:
            tokens += encoding.encode(chunk)
    elif isinstance(text, str):
        tokens = encoding.encode(text)
    else:
        raise ValueError("Input must be a list of strings or a single string.")

    response = client.embeddings.create(
        input=tokens,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def _question_to_context(query: str):
    vector = get_embeddings(query)
    query_results = query_index(vector)

    if not query_results or not query_results.matches:
        return {"question": query, "context": "No relevant context found."}

    context = " ".join([m.metadata.get("text", "") for m in query_results.matches])

    return {"question": query, "context": context}


def get_response(question: str) -> str:
    context_step = RunnableLambda(_question_to_context)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided context. If the context does not contain relevant information, respond with "No relevant information found".

    Context: {context}

    Question: {question}
    """)

    chain = context_step | prompt | llm

    response = chain.invoke(question)
    return response.content
