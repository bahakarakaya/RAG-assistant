from openai import OpenAI
import tiktoken
import numpy as np
from config import OPENAI_API_KEY
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from pinecone_manager import query_index

client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0.0)
embedding_func = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")


def load_doc(file_path: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    doc_data = loader.load()

    doc_data_str = ""
    for page in doc_data:
        doc_data_str += page.page_content

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=768,
        chunk_overlap=76
    )

    return splitter.split_text(doc_data_str)


def get_embeddings(text: str | list[str]) -> list[float]:
    model = "text-embedding-3-small"
    encoding = tiktoken.encoding_for_model(model)
    MAX_TOKENS = 8191

    if isinstance(text, list):
        text = " ".join(text)
    elif not isinstance(text, str):
        raise ValueError("Input must be a list of strings or a single string.")

    tokens = encoding.encode(text)
    print(len(tokens))
    if len(tokens) > MAX_TOKENS:
        all_embeddings = []

        for i in range(0, len(tokens), MAX_TOKENS):
            chunk_tokens = tokens[i:i + MAX_TOKENS]

            response = client.embeddings.create(
                input=chunk_tokens,
                model=model
            )
            embedding = response.data[0].embedding
            all_embeddings.append(embedding)

        mean_embedding = np.mean(all_embeddings, axis=0)
        return mean_embedding.tolist()

    else:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding


def _question_to_context(query: str):
    vector = get_embeddings(query)
    query_results = query_index(vector)

    if not query_results or not query_results.matches:
        return {"question": query, "context": "No relevant context found."}

    context = " ".join([m.metadata.get("text", "") for m in query_results.matches])
    for m in query_results.matches:
        print(m.metadata.get("text", "") + "\n\n")
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
