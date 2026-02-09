import streamlit as st
import chromadb
import os
import shutil
from git import Repo
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from huggingface_hub import InferenceClient

# --- Constants ---
TEMP_DIR = "./temp_repo"
DB_PATH = "./repo_db"
COLLECTION_NAME = "repo_code_collection"
SUPPORTED_EXTENSIONS = ('.py', '.js', '.java', '.cpp', '.c', '.rb', '.go', '.ts', '.html', '.css', '.md')
EXT_TO_LANG = {
    '.py': Language.PYTHON,
    '.js': Language.JAVASCRIPT,
    '.java': Language.JAVA,
    '.cpp': Language.CPP,
    '.c': Language.C,
    '.rb': Language.RUBY,
    '.go': Language.GO,
    '.ts': Language.TS,
    '.html': Language.HTML,
    '.md': Language.MARKDOWN
}

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# --- Model Loading (Cached) ---

def query_deepseek(prompt):
    """Sends a prompt to the DeepSeek model and returns the response."""
    try:
        messages = [{"role": "user", "content": prompt}]
        client = InferenceClient(token=os.environ["HF_TOKEN"])
        response = client.chat_completion(
            model= "deepseek-ai/DeepSeek-V3.2",
            messages=messages, 
            temperature=0.1
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"An unexpected error occurred while calling DeepSeek: {e}")
        st.error(f"An unexpected error occurred while calling DeepSeek: {e}")
        return None
    
def get_splitter_for_file(file_name):
    file_extension = os.path.splitext(file_name)[1]
    language = EXT_TO_LANG.get(file_extension)
    if language:
        return RecursiveCharacterTextSplitter.from_language(language, chunk_size=2000, chunk_overlap=200)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

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

    
    for i, file_path in enumerate(files_to_index):
        text_splitter = get_splitter_for_file(file_path)
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
            # Use one consistent, sorted list for both index finding and the widget
            sorted_files = sorted(st.session_state.indexed_files)

            # Determine the index based on the current session state
            if st.session_state.selected_file in sorted_files:
                current_index = sorted_files.index(st.session_state.selected_file)
            else:
                current_index = 0

            # The 'key' ensures Streamlit tracks this widget's state automatically
            st.session_state.selected_file = st.selectbox(
                "Select a file to inspect:", 
                sorted_files,
                index=current_index,
                key="file_selector"
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
                file_locator = {"file_path": st.session_state.selected_file} if st.session_state.selected_file else {}
                results = collection.query(query_texts=[prompt], n_results=5, where=file_locator)
                context = "\n\n---\n\n".join(results["documents"][0]) if results["documents"] else ""
                
                if context:
                    full_prompt = f"Based on the following context from files ({st.session_state.selected_file}), answer the user's question.\n\nContext:\n{context}\n\nQuestion: {prompt}\nAnswer:"
                    # Pass the cached model and tokenizer to the query function
                    response = query_deepseek(full_prompt)
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

