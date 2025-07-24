import streamlit as st
from streamlit_autorefresh import st_autorefresh
from config import OPENAI_API_KEY, namespace
from pinecone_manager import pc, pc_create_index_if_not_exists, upsert_to_pinecone, query_index
from rag_engine import load_doc, get_embeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import
import uuid
import time
import tempfile


from config import PINECONE_INDEX

#TODO: auto-delete namespace
# MAX_IDLE_SECONDS = 300
#
# st_autorefresh(interval=10 * 1000, key="autorefresh")
#
# random_id = str(uuid.uuid4())[:8]
# st.session_state["namespace"] = f"session_{random_id}"
#
# st.session_state["last_active"] = time.time()
#
# st.session_state["last_active"] = time.time()
# idle_time = time.time() - st.session_state["last_active"]
#
# if idle_time > MAX_IDLE_SECONDS:
#     pc.Index(PINECONE_INDEX).delete_namespace(st.session_state["namespace"])





st.title("RAG Assistant", anchor=None, help=None, width="stretch")
st.markdown("Q-A Assistant answers based on documents you uploaded. Only :rainbow[**PDF**] files can be uploaded", unsafe_allow_html=False, help=None, width="stretch")


uploaded_file = st.file_uploader("Choose a file", type="pdf",accept_multiple_files=False)
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    print("1")
    pc_create_index_if_not_exists()
    print("2")
    chunks = load_doc(temp_file_path)
    print("3")
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = get_embeddings(batch)
        upsert_to_pinecone([(str(i), embeddings) for i in range(len(batch))])

    st.success("File uploaded and processed successfully!", icon="✅")


#llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
#llm.stream()
#system_prompt = """."""

query = st.chat_input("Ask your question:")