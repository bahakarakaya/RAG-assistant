import streamlit as st
from streamlit_autorefresh import st_autorefresh
from config import OPENAI_API_KEY
from pinecone_manager import pc
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import
import uuid
import time

from rag_app.config import PINECONE_INDEX

#auto-delete namespace
MAX_IDLE_SECONDS = 300

st_autorefresh(interval=10 * 1000, key="autorefresh")

if "namespace" not in st.session_state:
    random_id = str(uuid.uuid4())[:8]
    st.session_state["namespace"] = f"session_{random_id}"

    st.session_state["last_active"] = time.time()

st.session_state["last_active"] = time.time()
idle_time = time.time() - st.session_state["last_active"]

if idle_time > MAX_IDLE_SECONDS:
    pc.Index(PINECONE_INDEX).delete_namespace(st.session_state["namespace"])





namespace = st.session_state["namespace"]

st.title("RAG Assistant", anchor=None, help=None, width="stretch")
st.markdown("Q-A Assistant answers based on documents you uploaded. Only :rainbow[**PDF**] files can be uploaded", unsafe_allow_html=False, help=None, width="stretch")

uploaded_file = st.file_uploader("Choose a file", type="pdf",accept_multiple_files=False)

#llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
#llm.stream()
#system_prompt = """."""

query = st.chat_input("Ask your question:")