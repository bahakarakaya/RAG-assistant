import streamlit as st
from pinecone_manager import pc_create_index_if_not_exists, upsert_to_pinecone
from rag_engine import load_doc, get_embeddings, get_response, embedding_func
import tempfile


st.title("RAG Assistant", anchor=None, help=None, width="stretch")
st.markdown("Q-A Assistant answers based on documents you uploaded. Only :rainbow[**PDF**] files can be uploaded", unsafe_allow_html=False, help=None, width="stretch")


uploaded_file = st.file_uploader("Choose a file", type="pdf", accept_multiple_files=False)
if uploaded_file is not None and not st.session_state.get("file_processed", False):
    with st.spinner("Processing your file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        pc_create_index_if_not_exists()

        chunks = load_doc(temp_file_path)
        embeddings_list = embedding_func.embed_documents(chunks)

        vectors_to_upsert = [(str(i), embeddings_list[i], {'text': chunks[i]}) for i in range(len(chunks))]
        upsert_to_pinecone(vectors_to_upsert)

    st.success(f"File processed successfully! {len(chunks)} chunks were indexed.")
    st.session_state["file_processed"] = True

if uploaded_file is None:
    st.session_state["file_processed"] = False
    if "messages" in st.session_state:
        del st.session_state.messages

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Question about the document"):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Generating response..."):
            response = get_response(query)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
