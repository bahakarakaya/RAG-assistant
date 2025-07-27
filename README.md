# RAG Q&A Assistant with PDF Support

This project implements a Retrieval-Augmented Generation (RAG) assistant that can answer questions based on the content of a PDF file that you upload. The application is built with a Streamlit interface, utilizing OpenAI for language modeling and embeddings, Langchain to streamline the creation of the RAG pipeline and Pinecone for efficient vector storage and retrieval.

#### Follow the link for live demo: https://rag-assistant-baa.streamlit.app/

## Technologies Used

*   **Streamlit:** For creating the interactive web application.
*   **OpenAI API:** For generating text embeddings and for the language model (`gpt-4.1-mini`).
*   **Pinecone:** For a high-performance vector database.
*   **LangChain:** To streamline the creation of the RAG pipeline.
*   **tiktoken:** For tokenizing text to be compatible with OpenAI models.

## Features

*   **PDF Document Upload:** Easily upload your PDF files through a user-friendly web interface.
*   **Intelligent Q&A:** Ask questions in natural language and receive answers generated from the content of your uploaded document.
*   **Chat History:** Keeps track of the current conversation.
*   **Efficient Document Processing:** The application processes your documents by splitting them into manageable chunks.
*   **Advanced Embeddings:** Utilizes OpenAI's powerful `text-embedding-3-small` model to generate vector embeddings.
*   **High-Speed Vector Search:** Employs Pinecone's serverless vector database for quick and scalable similarity searches.

## How It Works

The application follows a RAG architecture:

1.  **Document Upload and Processing:** When a PDF is uploaded, it's divided into smaller text chunks.
2.  **Embedding Generation:** Each text chunk is converted into a numerical representation (vector embedding) using OpenAI's embedding model.
3.  **Vector Storage:** These embeddings are stored in a Pinecone index for efficient retrieval.
4.  **Question Answering:** When you ask a question, it is also converted into an embedding. Pinecone then searches the index for the most relevant text chunks based on semantic similarity.
5.  **Response Generation:** The retrieved text chunks are provided as context to an OpenAI language model, which generates a comprehensive answer to your question.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the necessary dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a `config.py` file and add your API keys:
    ```python
    OPENAI_API_KEY = "your_openai_api_key"
    PINECONE_API_KEY = "your_pinecone_api_key"
    PINECONE_REGION = "your_pinecone_region"
    PINECONE_INDEX = "your_pinecone_index_name"
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run interface.py
    ```

2.  **Upload a PDF File:**
    Use the file uploader in the web interface to select and upload a PDF document.

3.  **Ask a Question:**
    Once the file has been processed, you can ask questions about the document's content in the chat input field. The assistant will then provide an answer based on the information in the document.