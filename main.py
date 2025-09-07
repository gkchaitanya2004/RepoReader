import streamlit as st
import subprocess
import chromadb
import os
import shutil
from git import Repo
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Constants ---
TEMP_DIR = "./temp_repo"
DB_PATH = "./repo_db"
COLLECTION_NAME = "repo_code_collection"
SUPPORTED_EXTENSIONS = ('.py', '.js', '.java', '.cpp', '.c', '.rb', '.go', '.ts', '.html', '.css', '.md')

# --- Model Loading (Cached) ---

def query_ollama(prompt, model="codellama:7b"):
    """Sends a prompt to the Ollama model and returns the response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        st.error("🚨 Ollama not found. Please ensure the 'ollama' command is in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"Error from Ollama: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while calling Ollama: {e}")
        return None

def process_repository(repo_url, collection):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    Repo.clone_from(repo_url, TEMP_DIR)

    files_to_index = []
    for root, _, filenames in os.walk(TEMP_DIR):
        for filename in filenames:
            if filename.endswith(SUPPORTED_EXTENSIONS):
                relative_path = os.path.relpath(os.path.join(root, filename), TEMP_DIR)
                files_to_index.append(relative_path)

    if not files_to_index:
        st.warning("No supported code files found in the repository.")
        return []

    st.info(f"Found {len(files_to_index)} files to index...")
    progress_bar = st.progress(0, text="Starting indexing...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    for i, file_path in enumerate(files_to_index):
        full_path = os.path.join(TEMP_DIR, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            chunks = text_splitter.split_text(content)
            if chunks:
                chunk_ids = [f"{file_path}-{j}" for j in range(len(chunks))]
                collection.add(
                    documents=chunks,
                    metadatas=[{"file_path": file_path}] * len(chunks),
                    ids=chunk_ids
                )
        except Exception as e:
            st.error(f"Error reading or indexing {file_path}: {e}")
        
        progress_bar.progress((i + 1) / len(files_to_index), text=f"Indexing {file_path}")

    progress_bar.empty()
    return files_to_index

# --- ChromaDB Setup ---
chroma_client = chromadb.PersistentClient(path=DB_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="RepoRover")
st.title("RepoRover: The AI Codebase Navigator 🚀")
st.subheader("Analyze and Query Your GitHub Repositories with AI")
st.markdown("Enter a public GitHub repository URL to start exploring its codebase with the help of AI.")


# --- Session State Initialization ---
if "processed_repo" not in st.session_state:
    st.session_state.processed_repo = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# --- Sidebar for Repository Input ---
with st.sidebar:
    st.header("Repository Configuration")
    github_link = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")

    if st.button("Analyze Repository", type="primary"):
        if github_link:
            if github_link != st.session_state.processed_repo:
                try:
                    chroma_client.delete_collection(name=COLLECTION_NAME)
                except Exception:
                    pass # Collection might not exist, which is fine.
                
                with st.spinner('Cloning and indexing repository... This may take a moment.'):
                    collection = chroma_client.get_or_create_collection(
                        name=COLLECTION_NAME, embedding_function=embedding_model
                    )
                    
                    indexed_files = process_repository(github_link, collection)
                    if indexed_files:
                        st.session_state.processed_repo = github_link
                        st.session_state.indexed_files = indexed_files
                        st.session_state.messages = {}
                        st.session_state.selected_file = None
                        st.success("Repository indexed successfully!")
                        st.rerun()
            else:
                st.info("This repository has already been analyzed.")
        else:
            st.warning("Please enter a GitHub repository URL.")

# --- Main Content Area ---
if not st.session_state.processed_repo:
    st.info("Enter a GitHub repository URL in the sidebar and click 'Analyze Repository' to begin.")
else:
    st.success(f"Currently analyzing: **{st.session_state.processed_repo}**")
    
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_model)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Explore Files")
        if st.session_state.indexed_files:
            current_selection = st.session_state.get('selected_file')
            index = 0
            if current_selection and current_selection in st.session_state.indexed_files:
                index = st.session_state.indexed_files.index(current_selection)

            st.session_state.selected_file = st.selectbox(
                "Select a file to inspect:", 
                sorted(st.session_state.indexed_files),
                index=index
            )

            if st.session_state.selected_file:
                full_path = os.path.join(TEMP_DIR, st.session_state.selected_file)
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    st.code(content, language='autodetect', line_numbers=True)

    with col2:
        st.subheader("Chat with your Codebase")
        repo_chat_key = "repo_chat_history"
        if repo_chat_key not in st.session_state.messages:
            st.session_state.messages[repo_chat_key] = []

        prompt = st.chat_input("Ask a question about the repository...")

        if prompt:
            st.session_state.messages[repo_chat_key].append({"role": "user", "content": prompt})
            with st.spinner("Searching the repository and generating a response..."):
                results = collection.query(query_texts=[prompt], n_results=5)
                context = "\n\n---\n\n".join(results["documents"][0]) if results["documents"] else ""
                
                if context:
                    file_paths = ", ".join(set(meta['file_path'] for meta in results['metadatas'][0]))
                    full_prompt = f"Based on the following context from files ({file_paths}), answer the user's question.\n\nContext:\n{context}\n\nQuestion: {prompt}\nAnswer:"
                    # Pass the cached model and tokenizer to the query function
                    response = query_ollama(full_prompt)
                    st.session_state.messages[repo_chat_key].append({"role": "assistant", "content": response, "context": context})
                else:
                    st.session_state.messages[repo_chat_key].append({"role": "assistant", "content": "I couldn't find any relevant context in the repository to answer your question.", "context": "No context found."})
            st.rerun()

        repo_chat_container = st.container(height=550)
        for message in st.session_state.messages[repo_chat_key]:
            with repo_chat_container.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("Show Retrieved Context"):
                        st.text(message["context"])

