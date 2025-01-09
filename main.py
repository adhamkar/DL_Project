import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from gtts import gTTS
import os
import tempfile
import base64


# Function to autoplay audio
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)


# Set Streamlit page configuration
st.set_page_config(page_title="PDF-RAG", page_icon="👀")

# Create a temporary directory for audio files if it doesn't exist
if 'audio_dir' not in st.session_state:
    st.session_state.audio_dir = tempfile.mkdtemp()

# Sidebar for PDF upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Main area for Q&A
st.title("Local PDF-RAG with LangChain, Ollama, and Chroma")

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Handle PDF upload and processing
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split the PDF
        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and store in Chroma
        embeddings = HuggingFaceEmbeddings()
        st.session_state.db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_store")
        st.session_state.db.persist()

        # Initialize the QA chain with Ollama
        llm = Ollama(model="mistral")  # Or your preferred Ollama model
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.db.as_retriever(),
            return_source_documents=True,
        )

        st.success("PDF processed and ready for questions!")

# User input for query
query = st.chat_input("Ask a question about the PDF:")

# Generate and display response
if query and st.session_state.qa_chain:
    with st.spinner("Generating response..."):
        result = st.session_state.qa_chain(query)

        # Create text-to-speech audio
        tts = gTTS(text=result["result"], lang='en')

        # Generate a unique filename
        audio_file = os.path.join(st.session_state.audio_dir, f"response_{hash(query)}.mp3")

        # Save and autoplay the audio file
        tts.save(audio_file)
        autoplay_audio(audio_file)


# Cleanup old audio files
def cleanup_old_files():
    if hasattr(st.session_state, 'audio_dir'):
        files = os.listdir(st.session_state.audio_dir)
        if len(files) > 5:  # Keep only the 5 most recent files
            files.sort(key=lambda x: os.path.getctime(os.path.join(st.session_state.audio_dir, x)))
            for old_file in files[:-5]:
                try:
                    os.remove(os.path.join(st.session_state.audio_dir, old_file))
                except:
                    pass


# Run cleanup periodically
cleanup_old_files()