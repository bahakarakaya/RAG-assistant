import os
import uuid
from dotenv import load_dotenv
from streamlit import session_state, secrets

load_dotenv()
if os.getenv("OPENAI_API_KEY"):  # for local development
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_REGION = os.getenv("PINECONE_ENV")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
else:  # for streamlit cloud deployment
    OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = secrets["PINECONE_API_KEY"]
    PINECONE_REGION = secrets["PINECONE_ENV"]
    PINECONE_INDEX = secrets["PINECONE_INDEX_NAME"]

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. " +
          "Please set it in your environment variables or .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set." +
          "Please set it in your environment variables or .env file.")

session_state["namespace"] = None
namespace = f"session_{str(uuid.uuid4())[:6]}"
session_state["namespace"] = namespace
