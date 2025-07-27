import streamlit as st
from pinecone_manager import pc_create_index_if_not_exists, upsert_to_pinecone
from rag_engine import load_doc, get_embeddings, get_response
import tempfile


st.title("RAG Assistant", anchor=None, help=None, width="stretch")
st.markdown("Q-A Assistant answers based on documents you uploaded. Only :rainbow[**PDF**] files can be uploaded", unsafe_allow_html=False, help=None, width="stretch")


uploaded_file = st.file_uploader("Choose a file", type="pdf", accept_multiple_files=False)
if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        pc_create_index_if_not_exists()

        chunks = load_doc(temp_file_path)
        embeddings = get_embeddings(chunks)

        upsert_to_pinecone([(str(i), embeddings, {'text': chunks[i]}) for i in range(len(chunks))])

    file_processed = True

if "messages" not in st.session_state:
    st.session_state.messages = []

# Accept user input
if uploaded_file:
    if query := st.chat_input("Question about the document"):
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

        # Display user message in chat message container
        with st.chat_message("user"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            st.markdown(query)

        with st.spinner("Generating response..."):
            response = get_response(query)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
