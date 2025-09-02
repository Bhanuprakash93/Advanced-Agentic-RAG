import streamlit as st
import base64
import time
import fitz  # PyMuPDF
import pickle
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from openai import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from typing import List, Optional, Dict, Any
import requests
import json

# DeepSeek configuration
DEEPSEEK_CONFIG = {
    "DEEPSEEK_API_KEY": "",
    "DEEPSEEK_API_URL": "https://api.deepseek.com/v1/chat/completions"
}

# Custom DeepSeek Reranker implementation
class DeepSeekReranker(BaseDocumentCompressor):
    """Custom reranker using DeepSeek LLM to reorder documents by relevance."""
    
    def __init__(self, top_n: int = 3):
        self.top_n = top_n
    
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Rerank documents based on relevance to the query using DeepSeek."""
        if not documents or not query:
            return documents[:self.top_n]
        
        print(f"DeepSeek reranking {len(documents)} documents to top {self.top_n}")
        
        try:
            # Create prompt for relevance scoring
            relevance_prompt = f"""
            You are an expert document relevance evaluator. Given a question and a list of document excerpts, 
            evaluate how relevant each document is to answering the question.

            Question: {query}

            Documents to evaluate:
            {self._format_documents_for_evaluation(documents)}

            Please evaluate each document's relevance to the question on a scale of 0.0 to 1.0, where:
            - 1.0: Perfectly relevant, directly answers the question
            - 0.5: Somewhat relevant, contains related information
            - 0.0: Completely irrelevant to the question

            Return your evaluation as a JSON object with document indices as keys and relevance scores as values.
            Example: {{"0": 0.8, "1": 0.3, "2": 0.9}}

            Only return the JSON object, nothing else.
            """
            
            # Call DeepSeek API for relevance scoring
            response = call_deepseek_api(relevance_prompt, max_tokens=2000)
            
            if response:
                # Parse the relevance scores
                relevance_scores = self._parse_relevance_scores(response, len(documents))
                
                # Add relevance scores to document metadata
                for i, doc in enumerate(documents):
                    doc.metadata['relevance_score'] = relevance_scores.get(str(i), 0.0)
                    doc.metadata['original_score'] = i  # Store original ranking
                
                # Sort documents by relevance score (descending)
                sorted_docs = sorted(documents, key=lambda x: x.metadata.get('relevance_score', 0.0), reverse=True)
                
                print(f"DeepSeek reranking completed. Top scores: {[doc.metadata.get('relevance_score', 0.0) for doc in sorted_docs[:3]]}")
                
                return sorted_docs[:self.top_n]
            
        except Exception as e:
            print(f"DeepSeek reranking failed: {e}")
        
        # Fallback: return top documents without reranking
        return documents[:self.top_n]
    
    def _format_documents_for_evaluation(self, documents: List[Document]) -> str:
        """Format documents for the evaluation prompt."""
        formatted = []
        for i, doc in enumerate(documents):
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted.append(f"Document {i}:\n{content_preview}\n")
        return "\n".join(formatted)
    
    def _parse_relevance_scores(self, response: str, num_documents: int) -> Dict[str, float]:
        """Parse relevance scores from DeepSeek response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                # Validate and normalize scores
                validated_scores = {}
                for key, value in scores.items():
                    if key.isdigit() and int(key) < num_documents:
                        validated_scores[key] = max(0.0, min(1.0, float(value)))
                return validated_scores
        except Exception as e:
            print(f"Failed to parse relevance scores: {e}")
        
        # Fallback: return equal scores
        return {str(i): 0.5 for i in range(num_documents)}
    
    async def acompress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Async version of compress_documents (not implemented)."""
        return self.compress_documents(documents, query)

# Reranker availability flag
DEEPSEEK_RERANKER_AVAILABLE = True

# Configuration - Replace with your actual Azure OpenAI configuration
AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": "",
    "AZURE_OPENAI_API_KEY": "",
    "OPENAI_API_VERSION": "",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": ""
}

# Directory for storing vector databases and RAGAS cache
VECTOR_DB_DIR = "vector_databases"
METADATA_FILE = "db_metadata.json"
RAGAS_CACHE_DIR = "ragas_cache"

# CSS Styling (kept same as original)
st.markdown("""
<style>
.header-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
}
.welcome-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}
.welcome-title {
    color: #2c3e50;
    margin-bottom: 1rem;
}
.welcome-text {
    color: #34495e;
    line-height: 1.6;
}
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.feature-item {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
.feature-desc {
    color: #7f8c8d;
    font-size: 0.9rem;
}
.stats-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.stat-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    flex: 1;
}
.stat-number {
    font-size: 2rem;
    font-weight: bold;
}
.stat-label {
    font-size: 0.8rem;
    opacity: 0.9;
}
.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.section-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #667eea;
}
.main-content {
    min-height: 400px;
}
</style>
""", unsafe_allow_html=True)

def ensure_directories():
    """Ensure necessary directories exist"""
    Path(VECTOR_DB_DIR).mkdir(exist_ok=True)
    Path(RAGAS_CACHE_DIR).mkdir(exist_ok=True)

def validate_azure_config():
    """Validate Azure configuration"""
    required_fields = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
    ]
    
    for field in required_fields:
        if not AZURE_CONFIG.get(field):
            st.error(f"Missing Azure configuration: {field}")
            return False
    return True

def get_cache_key(question: str, answer: str, contexts: List[str]) -> str:
    """Generate a cache key for RAGAS evaluation"""
    content = f"{question}|{answer}|{'|'.join(contexts)}"
    return hashlib.md5(content.encode()).hexdigest()

def save_ragas_cache(cache_key: str, metrics: Dict):
    """Save RAGAS metrics to cache"""
    try:
        cache_file = os.path.join(RAGAS_CACHE_DIR, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Failed to save RAGAS cache: {e}")

def load_ragas_cache(cache_key: str) -> Optional[Dict]:
    """Load RAGAS metrics from cache"""
    cache_file = os.path.join(RAGAS_CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load RAGAS cache: {e}")
    return None

def get_metric_color(score: float) -> str:
    """Get color based on metric score"""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"

def call_deepseek_api(prompt: str, max_tokens: int = 1000) -> str:
    """Call DeepSeek API with the given prompt"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_CONFIG['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        response = requests.post(
            DEEPSEEK_CONFIG["DEEPSEEK_API_URL"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"DeepSeek API error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return ""

def evaluate_with_deepseek(question: str, answer: str, contexts: List[str]) -> Dict:
    """Evaluate response using DeepSeek LLM for faithfulness and relevance."""
    ensure_directories()
    
    cache_key = get_cache_key(question, answer, contexts)
    cached_metrics = load_ragas_cache(cache_key)
    if cached_metrics:
        print(f"Using cached metrics: {cached_metrics}")
        return cached_metrics
    
    print("Starting DeepSeek evaluation...")
    
    try:
        # Prepare context for evaluation
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)]) if contexts else "No context available"
        
        # Faithfulness evaluation prompt
        faithfulness_prompt = f"""
        Evaluate the faithfulness of the answer to the provided context. Faithfulness measures whether the answer is factually consistent with the context.

        Question: {question}
        Answer: {answer}
        Context: {context_text}

        Please analyze and provide a score between 0.0 and 1.0 where:
        - 1.0: The answer is completely faithful to the context (all facts are supported)
        - 0.0: The answer contains information not present in the context or contradicts the context

        Provide your response in JSON format with a single "faithfulness_score" field containing the numeric score.
        """
        
        # Answer relevance evaluation prompt
        relevance_prompt = f"""
        Evaluate the relevance of the answer to the question. Relevance measures how well the answer addresses the specific question.

        Question: {question}
        Answer: {answer}
        Context: {context_text}

        Please analyze and provide a score between 0.0 and 1.0 where:
        - 1.0: The answer perfectly addresses the question and is highly relevant
        - 0.0: The answer is completely irrelevant to the question

        Provide your response in JSON format with a single "relevance_score" field containing the numeric score.
        """
        
        # Get faithfulness score
        faithfulness_response = call_deepseek_api(faithfulness_prompt)
        faithfulness_score = 0.0
        if faithfulness_response:
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', faithfulness_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    faithfulness_score = float(result.get("faithfulness_score", 0.0))
                else:
                    # Fallback: look for numeric score in text
                    score_match = re.search(r'(\d+\.\d+)', faithfulness_response)
                    if score_match:
                        faithfulness_score = float(score_match.group(1))
            except:
                faithfulness_score = 0.0
        
        # Get relevance score
        relevance_response = call_deepseek_api(relevance_prompt)
        relevance_score = 0.0
        if relevance_response:
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    relevance_score = float(result.get("relevance_score", 0.0))
                else:
                    # Fallback: look for numeric score in text
                    score_match = re.search(r'(\d+\.\d+)', relevance_response)
                    if score_match:
                        relevance_score = float(score_match.group(1))
            except:
                relevance_score = 0.0
        
        metrics = {
            "faithfulness": max(0.0, min(1.0, faithfulness_score)),
            "answer_relevancy": max(0.0, min(1.0, relevance_score)),
        }
        
        print(f"FINAL DEEPSEEK METRICS: {metrics}")
        
        save_ragas_cache(cache_key, metrics)
        print("Metrics cached successfully")
        
        return metrics
        
    except Exception as e:
        error_msg = f"DeepSeek evaluation failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

def save_vector_db_metadata(db_name: str, file_names: List[str], stats: Dict):
    """Save metadata about the vector database"""
    ensure_directories()
    metadata_path = os.path.join(VECTOR_DB_DIR, METADATA_FILE)
    
    # Load existing metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            metadata = {}
    
    # Add new entry
    metadata[db_name] = {
        "created_at": datetime.now().isoformat(),
        "file_names": file_names,
        "stats": stats,
        "total_files": len(file_names)
    }
    
    # Save updated metadata
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Failed to save metadata: {e}")

def load_vector_db_metadata() -> Dict:
    """Load vector database metadata"""
    metadata_path = os.path.join(VECTOR_DB_DIR, METADATA_FILE)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
    return {}

def save_vector_db(vectorstore, db_name: str, document_sources: Dict, uploaded_files_data: Dict = None):
    """Save vector database with improved error handling"""
    ensure_directories()
    db_path = os.path.join(VECTOR_DB_DIR, f"{db_name}")
    sources_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_sources.pkl")
    files_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_files.pkl")
    
    try:
        # Save vector database
        vectorstore.save_local(db_path)
        
        # Save document sources mapping
        with open(sources_path, 'wb') as f:
            pickle.dump(document_sources, f)
        
        # Save uploaded files data for later retrieval
        if uploaded_files_data:
            with open(files_path, 'wb') as f:
                pickle.dump(uploaded_files_data, f)
                
        return True
    except Exception as e:
        st.error(f"Failed to save vector database: {e}")
        return False

def load_vector_db(db_name: str, embeddings) -> Tuple[Optional[FAISS], Optional[Dict], Optional[Dict]]:
    """Load vector database with improved error handling"""
    db_path = os.path.join(VECTOR_DB_DIR, f"{db_name}")
    sources_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_sources.pkl")
    files_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_files.pkl")
    
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            st.error(f"Database path does not exist: {db_path}")
            return None, None, None
            
        # Load vector database
        vectorstore = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Load document sources
        document_sources = {}
        if os.path.exists(sources_path):
            with open(sources_path, 'rb') as f:
                document_sources = pickle.load(f)
        
        # Load uploaded files data
        uploaded_files_data = {}
        if os.path.exists(files_path):
            with open(files_path, 'rb') as f:
                uploaded_files_data = pickle.load(f)
        
        return vectorstore, document_sources, uploaded_files_data
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None, None, None

def display_pdf_viewer(file_name: str, start_page: int, response_index: int):
    """PDF viewer with unique keys and navigation controls below the image."""
    if file_name not in st.session_state.get('uploaded_file_bytes', {}):
        st.error(f"File '{file_name}' not found in current session.")
        return
    
    file_bytes = st.session_state.uploaded_file_bytes[file_name]
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # MODIFIED: Create a unique key for the page state
        page_state_key = f"current_page_{file_name}_{response_index}"

        st.markdown(f"**{file_name}** - Page {start_page} of {total_pages}")
        
        # MODIFIED: Display the PDF image first
        page = doc[start_page - 1]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, use_container_width=True)
        
        doc.close()

        # MODIFIED: Place navigation controls and columns below the image
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("◀ Previous", disabled=(start_page <= 1), key=f"prev_{file_name}_{response_index}"):
                st.session_state[page_state_key] = max(1, start_page - 1)
                st.rerun()
        
        with col2:
            if st.button("Next ▶", disabled=(start_page >= total_pages), key=f"next_{file_name}_{response_index}"):
                st.session_state[page_state_key] = min(total_pages, start_page + 1)
                st.rerun()
        
        with col3:
            target_page = st.number_input(
                f"Go to page (1-{total_pages})", 
                min_value=1, max_value=total_pages, value=start_page,
                key=f"page_input_{file_name}_{response_index}"
            )
            # Add a button to jump, or make it react on change
            if st.button("Jump", key=f"jump_{file_name}_{response_index}"):
                if target_page != start_page:
                    st.session_state[page_state_key] = target_page
                    st.rerun()

    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def display_header():
    """Display professional header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">PDF Document Assistant with Reranking</h1>
        <p class="subtitle">Upload and analyze your PDF documents and images with AI-powered document reranking</p>
    </div>
    """, unsafe_allow_html=True)

def display_welcome():
    """Display welcome section"""
    st.markdown("""
    <div class="welcome-section">
        <h2 class="welcome-title">Welcome to Enhanced PDF Document Assistant</h2>
        <p class="welcome-text">
            Upload your PDF documents and images to start analyzing and asking questions about their content. 
            The system uses advanced DeepSeek LLM reranking to provide the most relevant information first.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Multi-Format Support</div>
            <div class="feature-desc">Process PDFs and image files simultaneously</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🏆</div>
            <div class="feature-title">Smart Reranking</div>
            <div class="feature-desc">DeepSeek LLM reorders results by relevance</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💬</div>
            <div class="feature-title">AI Chat</div>
            <div class="feature-desc">Ask questions about your documents</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Quality Metrics</div>
            <div class="feature-desc">DeepSeek LLM evaluation for response quality</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_stats(stats):
    """Display processing statistics"""
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-number">{stats.get('text_pages', 0)}</div>
            <div class="stat-label">Text Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('ocr_pages', 0)}</div>
            <div class="stat-label">OCR Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('image_files', 0)}</div>
            <div class="stat-label">Image Files</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_reranker_status():
    """Display reranker configuration status in the sidebar."""
    st.markdown("---")
    st.markdown('<div class="section-title">🏆 Reranker Status</div>', unsafe_allow_html=True)
    
    if DEEPSEEK_RERANKER_AVAILABLE:
        st.success("✅ DeepSeek Reranker Available")
        st.info("📊 Reranking active:\n- Retrieves 10 documents\n- Reranks to top N using DeepSeek LLM\n- Shows ranking in results")
    else:
        st.error("❌ DeepSeek Reranker Not Available")
        st.warning("DeepSeek API configuration required")

def get_text_from_image(client: AzureOpenAI, page: fitz.Page, page_num: int, file_name: str) -> str:
    """Extracts text from a PDF page image using Azure's vision model."""
    try:
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        vision_prompt = "Extract all text from this image of a document page. Preserve the original formatting as accurately as possible."
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Rate limit hit during OCR. Retrying in {base_delay}s...")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    st.error(f"An unexpected error occurred during OCR: {e}")
                    return ""
        st.error("OCR failed after multiple retries.")
        return ""
    except Exception as e:
        st.error(f"Error in get_text_from_image: {e}")
        return ""

def get_text_from_uploaded_image(client: AzureOpenAI, image_file, file_name: str) -> str:
    """Extracts text from an uploaded image file using Azure's vision model."""
    try:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_file.seek(0)  # Reset file pointer

        vision_prompt = "Extract all text from this image of a document. Preserve the original formatting as accurately as possible."
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Rate limit hit during OCR. Retrying in {base_delay}s...")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    st.error(f"An unexpected error occurred during OCR: {e}")
                    return ""
        st.error("OCR failed after multiple retries.")
        return ""
    except Exception as e:
        st.error(f"Error in get_text_from_uploaded_image: {e}")
        return ""

def get_documents_text_with_sources(uploaded_files: list, client: AzureOpenAI) -> Tuple[List[Document], Dict, Dict]:
    """Extracts text from uploaded files with improved error handling"""
    documents = []
    processing_stats = {"text_pages": 0, "ocr_pages": 0, "image_files": 0}
    document_sources = {}
    OCR_THRESHOLD = 50

    for file in uploaded_files:
        try:
            if file.type == "application/pdf":
                file_content = file.read()
                file.seek(0)  # Reset for potential reuse
                
                doc = fitz.open(stream=file_content, filetype="pdf")
                for i, page in enumerate(doc):
                    page_text = page.get_text().strip()
                    page_num = i + 1
                    
                    if len(page_text) > OCR_THRESHOLD:
                        text_content = page_text
                        processing_stats["text_pages"] += 1
                    else:
                        st.write(f"Processing page {page_num} of '{file.name}' with OCR...")
                        text_content = get_text_from_image(client, page, page_num, file.name)
                        if text_content:
                            processing_stats["ocr_pages"] += 1
                    
                    if text_content and text_content.strip():
                        # Create Document with metadata
                        document = Document(
                            page_content=text_content,
                            metadata={
                                "source": file.name,
                                "page": page_num,
                                "type": "pdf"
                            }
                        )
                        documents.append(document)
                        
                        # Store source mapping for later retrieval
                        doc_id = len(documents) - 1
                        document_sources[doc_id] = {
                            "file_name": file.name,
                            "page": page_num,
                            "type": "pdf"
                        }
                doc.close()
            else:
                # Handle image files
                st.write(f"Processing image '{file.name}' with OCR...")
                text_content = get_text_from_uploaded_image(client, file, file.name)
                if text_content and text_content.strip():
                    processing_stats["image_files"] += 1
                    
                    # Create Document with metadata
                    document = Document(
                        page_content=text_content,
                        metadata={
                            "source": file.name,
                            "page": 1,
                            "type": "image"
                        }
                    )
                    documents.append(document)
                    
                    # Store source mapping
                    doc_id = len(documents) - 1
                    document_sources[doc_id] = {
                        "file_name": file.name,
                        "page": 1,
                        "type": "image"
                    }
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
    
    return documents, processing_stats, document_sources

def get_text_chunks_with_sources(documents: List[Document]) -> List[Document]:
    """Splits documents into manageable chunks while preserving metadata."""
    if not documents:
        return []
        
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1200, 
        chunk_overlap=200, 
        length_function=len
    )
    
    chunked_documents = []
    for doc in documents:
        if doc.page_content and doc.page_content.strip():
            try:
                # Split the document content
                chunks = text_splitter.split_text(doc.page_content)
                
                # Create new Document objects for each chunk, preserving metadata
                for chunk in chunks:
                    if chunk.strip():  # Only add non-empty chunks
                        chunked_doc = Document(
                            page_content=chunk,
                            metadata=doc.metadata.copy()
                        )
                        chunked_documents.append(chunked_doc)
            except Exception as e:
                st.error(f"Error chunking document: {e}")
    
    return chunked_documents

def get_vectorstore_from_documents(documents: List[Document]):
    """Creates a FAISS vector store from Document objects with improved error handling."""
    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_documents:
        st.error("Could not find any valid text content to process in the document(s).")
        return None

    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
        )

        st.write(f"Creating embeddings for {len(valid_documents)} document chunks...")
        vectorstore = None
        BATCH_SIZE = 16

        progress_bar = st.progress(0)
        
        try:
            for i in range(0, len(valid_documents), BATCH_SIZE):
                batch = valid_documents[i:i + BATCH_SIZE]
                
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vectorstore.add_documents(documents=batch)
                
                progress = min(((i + BATCH_SIZE) / len(valid_documents)), 1.0)
                progress_bar.progress(progress)
                time.sleep(1)
            
            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"An error occurred during embedding creation: {e}")
            return None

    except Exception as e:
        st.error(f"An error occurred during embedding initialization: {e}")
        return None

    return vectorstore

def get_conversation_chain_with_sources(vectorstore, faithfulness_threshold: float = 0.5, reranker_top_n: int = 10):
    """Enhanced conversation chain with configurable reranking to retrieve more sources."""
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
            api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
            temperature=0.0,
            model_version="latest"
        )
        
        # Create base retriever - retrieve more documents
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 15  # Retrieve even more documents for better reranking
            }
        )
        
        # Use DeepSeek reranker
        try:
            # Create DeepSeek reranker with configurable top_n
            compressor = DeepSeekReranker(top_n=reranker_top_n)
            
            # Create compression retriever with reranking
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            print(f"DeepSeek reranker successfully created! Will rerank top {reranker_top_n} from 15 retrieved documents.")
            
        except Exception as e:
            print(f"Failed to create DeepSeek reranker: {e}")
            print("Falling back to basic retriever without reranking")
            st.warning(f"Reranking failed, using basic retrieval: {e}")
            # Fallback to basic retriever with reduced k
            compression_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": reranker_top_n}
            )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=compression_retriever, 
            memory=memory,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def format_response_with_deepseek_metrics(response, question: str, faithfulness_threshold: float):
    """Format response with DeepSeek metrics and enhanced source information."""
    # Extract contexts and answer safely
    source_documents = response.get('source_documents', [])
    contexts = [doc.page_content for doc in source_documents if hasattr(doc, 'page_content')]
    answer = response.get('answer', '')
    
    # Run DeepSeek evaluation
    deepseek_metrics = evaluate_with_deepseek(question, answer, contexts)
    
    # Prepare enhanced source information with ranking details
    sources = []
    for i, doc in enumerate(source_documents):
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            source_info = {
                'rank': i + 1,
                'file_name': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'type': doc.metadata.get('type', 'pdf'),
                'relevance_score': getattr(doc, 'relevance_score', None),
                'original_score': getattr(doc, 'original_score', None),
            }
            sources.append(source_info)
    
    # Check if we should block the answer based on faithfulness threshold
    faithfulness_score = deepseek_metrics.get('faithfulness', 0.0) if deepseek_metrics else 0.0
    
    # Determine final answer to display
    if not source_documents:
        final_answer = "Could not find the Answer"
    elif faithfulness_score < faithfulness_threshold and faithfulness_score > 0.0:
        final_answer = "Could not find a reliable answer (faithfulness score too low)"
    else:
        final_answer = answer
    
    return final_answer, sources, deepseek_metrics


def display_all_sources_interactive(sources, response_index: int):
    """Display all retrieved sources with interactive viewing capabilities."""
    if not sources:
        st.info("No relevant sources were found for this response.")
        return
    
    st.markdown("##### Retrieved Sources")
    
    # Create tabs for each source
    if len(sources) == 1:
        # If only one source, display directly without tabs
        source = sources[0]
        display_single_source(source, response_index, 0)
    else:
        # Create tabs for multiple sources
        tab_names = []
        for i, source in enumerate(sources):
            file_name = source.get('file_name', 'Unknown')
            page_num = source.get('page', 1)
            rank = source.get('rank', i + 1)
            relevance_score = source.get('relevance_score', 0.0)
            
            # Create tab name with rank and relevance score if available
            if relevance_score is not None and relevance_score > 0:
                tab_name = f"#{rank} - {file_name[:15]}... (P{page_num}) [{relevance_score:.2f}]"
            else:
                tab_name = f"#{rank} - {file_name[:15]}... (P{page_num})"
            tab_names.append(tab_name)
        
        tabs = st.tabs(tab_names)
        
        for i, (tab, source) in enumerate(zip(tabs, sources)):
            with tab:
                display_single_source(source, response_index, i)

def display_single_source(source, response_index: int, source_index: int):
    """Display a single source with viewing capabilities."""
    file_name = source.get('file_name', 'Unknown')
    page_num = source.get('page', 1)
    source_type = source.get('type', 'pdf')
    rank = source.get('rank', source_index + 1)
    relevance_score = source.get('relevance_score', None)
    content = source.get('content', '')
    
    # Display source metadata
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**File:** {file_name}")
    with col2:
        st.markdown(f"**Page:** {page_num}")
    with col3:
        if relevance_score is not None and relevance_score > 0:
            color = "green" if relevance_score > 0.7 else "orange" if relevance_score > 0.4 else "red"
            st.markdown(f"**Score:** <span style='color: {color}'>{relevance_score:.2f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Rank:** #{rank}")
    
    # Display content preview
    if content:
        with st.expander("Content Preview", expanded=False):
            st.text(content)
    
    # Display file viewer if available
    if file_name in st.session_state.get('uploaded_file_bytes', {}):
        if source_type == 'pdf':
            # Create unique key for this source's page state
            page_state_key = f"current_page_{file_name}_{response_index}_{source_index}"
            if page_state_key not in st.session_state:
                st.session_state[page_state_key] = page_num
            
            current_page = st.session_state[page_state_key]
            
            # Create a button to view the PDF
            if st.button(f"View PDF", key=f"view_pdf_{response_index}_{source_index}"):
                st.session_state[f"show_pdf_{response_index}_{source_index}"] = True
            
            # Show PDF viewer if requested
            if st.session_state.get(f"show_pdf_{response_index}_{source_index}", False):
                display_pdf_viewer_for_source(file_name, current_page, response_index, source_index)
                
                # Add close button
                if st.button("Close PDF Viewer", key=f"close_pdf_{response_index}_{source_index}"):
                    st.session_state[f"show_pdf_{response_index}_{source_index}"] = False
                    st.rerun()
                    
        elif source_type == 'image':
            # Create a button to view the image
            if st.button(f"View Image", key=f"view_img_{response_index}_{source_index}"):
                st.session_state[f"show_img_{response_index}_{source_index}"] = True
            
            # Show image viewer if requested
            if st.session_state.get(f"show_img_{response_index}_{source_index}", False):
                file_bytes = st.session_state.uploaded_file_bytes[file_name]
                st.image(file_bytes, caption=f"Source Image: {file_name}", use_container_width=True)
                
                # Add close button
                if st.button("Close Image Viewer", key=f"close_img_{response_index}_{source_index}"):
                    st.session_state[f"show_img_{response_index}_{source_index}"] = False
                    st.rerun()
    else:
        st.warning(f"File '{file_name}' is not available for viewing in this session.")


def display_pdf_viewer_for_source(file_name: str, start_page: int, response_index: int, source_index: int):
    """PDF viewer specifically for individual sources with unique keys."""
    if file_name not in st.session_state.get('uploaded_file_bytes', {}):
        st.error(f"File '{file_name}' not found in current session.")
        return
    
    file_bytes = st.session_state.uploaded_file_bytes[file_name]
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # Create unique key for the page state
        page_state_key = f"current_page_{file_name}_{response_index}_{source_index}"

        st.markdown(f"**{file_name}** - Page {start_page} of {total_pages}")
        
        # Display the PDF image
        page = doc[start_page - 1]
        pix = page.get_pixmap(dpi=150)  # Reduced DPI for faster loading in tabs
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, use_container_width=True)
        
        doc.close()

        # Navigation controls below the image
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("◀ Previous", disabled=(start_page <= 1), key=f"prev_{file_name}_{response_index}_{source_index}"):
                st.session_state[page_state_key] = max(1, start_page - 1)
                st.rerun()
        
        with col2:
            if st.button("Next ▶", disabled=(start_page >= total_pages), key=f"next_{file_name}_{response_index}_{source_index}"):
                st.session_state[page_state_key] = min(total_pages, start_page + 1)
                st.rerun()
        
        with col3:
            target_page = st.number_input(
                f"Go to page (1-{total_pages})", 
                min_value=1, max_value=total_pages, value=start_page,
                key=f"page_input_{file_name}_{response_index}_{source_index}"
            )
            if st.button("Jump", key=f"jump_{file_name}_{response_index}_{source_index}"):
                if target_page != start_page:
                    st.session_state[page_state_key] = target_page
                    st.rerun()

    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

def display_interactive_rank_1_source(sources, response_index: int):
    """Shows the top-ranked source with smaller heading and less spacing."""
    if not sources:
        st.info("No relevant sources were found for this response.")
        return

    top_source = next((s for s in sources if s.get('rank') == 1), sources[0])
    
    file_name = top_source.get('file_name', 'Unknown')
    page_num = top_source.get('page', 1)
    source_type = top_source.get('type', 'pdf')

    # MODIFIED: Heading is now smaller (h5) for a more compact look
    st.markdown("##### Top Relevant Source")
    # MODIFIED: Removed the horizontal rule (st.markdown("---")) to decrease vertical spacing

    if file_name in st.session_state.get('uploaded_file_bytes', {}):
        if source_type == 'pdf':
            page_state_key = f"current_page_{file_name}_{response_index}"
            if page_state_key not in st.session_state:
                st.session_state[page_state_key] = page_num
            
            current_page = st.session_state[page_state_key]
            display_pdf_viewer(file_name, current_page, response_index)
            
        elif source_type == 'image':
            file_bytes = st.session_state.uploaded_file_bytes[file_name]
            st.image(file_bytes, caption=f"Source Image: {file_name}", use_container_width=True)
    else:
        st.warning(f"File '{file_name}' is not available for viewing in this session.")

    if len(sources) > 1:
        st.info(f"Found {len(sources) - 1} additional relevant sources.")
        
def display_ragas_metrics(ragas_metrics):
    """Display RAGAS metrics in a consistent format."""
    if not ragas_metrics:
        st.markdown("""
        <div style="border: 2px solid #f44336; padding: 10px; margin: 10px 0; background-color: #ffebee; border-radius: 5px;">
            <strong>RAGAS Evaluation Failed</strong><br>
            <span style="color: #666;">Unable to calculate response quality metrics</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    faithfulness_score = ragas_metrics.get('faithfulness', 0.0)
    relevancy_score = ragas_metrics.get('answer_relevancy', 0.0)
    
    faithfulness_color = get_metric_color(faithfulness_score)
    relevancy_color = get_metric_color(relevancy_score)
    
    # Determine overall quality indicator
    avg_score = (faithfulness_score + relevancy_score) / 2
    if avg_score >= 0.8:
        border_color = "#4CAF50"
        bg_color = "#f0f8f0"
        quality_text = "High Quality"
    elif avg_score >= 0.6:
        border_color = "#FF9800"
        bg_color = "#fff8e1"
        quality_text = "Medium Quality"
    else:
        border_color = "#f44336"
        bg_color = "#ffebee"
        quality_text = "Low Quality"
    
    st.markdown(f"""
    <div style="border: 2px solid {border_color}; padding: 15px; margin: 15px 0; background-color: {bg_color}; border-radius: 8px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <strong style="font-size: 16px;">Response Quality: {quality_text}</strong>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <strong>Faithfulness:</strong><br>
                <span style="color: {faithfulness_color}; font-weight: bold; font-size: 18px;">
                    {faithfulness_score:.1%}
                </span>
                <small style="display: block; color: #666;">Answer accuracy vs context</small>
            </div>
            <div>
                <strong>Answer Relevancy:</strong><br>
                <span style="color: {relevancy_color}; font-weight: bold; font-size: 18px;">
                    {relevancy_score:.1%}
                </span>
                <small style="display: block; color: #666;">How well answer addresses question</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def handle_user_input_with_deepseek(user_question: str, faithfulness_threshold: float):
    """Enhanced handler that shows ranking information."""
    if st.session_state.conversation:
        try:
            print(f"Processing question: {user_question}")
            
            # Use invoke instead of deprecated __call__
            response = st.session_state.conversation.invoke({'question': user_question})
            
            # Log retrieval info
            source_docs = response.get('source_documents', [])
            print(f"Retrieved {len(source_docs)} documents after reranking")
            
            for i, doc in enumerate(source_docs):
                if hasattr(doc, 'metadata'):
                    print(f"  Rank {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")
            
            answer, sources, ragas_metrics = format_response_with_deepseek_metrics(response, user_question, faithfulness_threshold)
            
            # Store both answer and sources in chat history
            st.session_state.chat_history = response.get('chat_history', [])
            
            # Store sources separately for display
            if 'response_sources' not in st.session_state:
                st.session_state.response_sources = []
            st.session_state.response_sources.append({
                'question': user_question,
                'sources': sources,
                'ragas_metrics': ragas_metrics
            })
            
            return answer, sources, ragas_metrics
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return "An error occurred while processing your question.", [], None
    else:
        st.warning("Please upload and process your files first or load a saved vector database.")
        return "", [], None

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="PDF Document Assistant with Reranking",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Inject CSS for the vertical line and column styling
    st.markdown("""
    <style>
    /* Simple vertical line separator */
    .vertical-divider {
        border-left: 2px solid #ddd;
        padding-left: 20px;
        margin-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Validate configuration at startup
    if not validate_azure_config():
        st.error("Please configure Azure OpenAI settings before using the application.")
        st.stop()

    # Display header
    display_header()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None
    if "document_sources" not in st.session_state:
        st.session_state.document_sources = {}
    if "response_sources" not in st.session_state:
        st.session_state.response_sources = []
    if "current_db_name" not in st.session_state:
        st.session_state.current_db_name = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = {}

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

        # Configuration settings
        st.markdown('<div class="section-title">Configuration Settings</div>', unsafe_allow_html=True)
        faithfulness_threshold = st.slider(
            "Faithfulness Threshold", 0.0, 1.0, 0.7, 0.05,
            help="Minimum faithfulness score required to provide an answer (RAGAS metric)"
        )
        reranker_top_n = st.slider(
            "Reranker Top N", 1, 10, 10, 1,
            help="Number of top documents to keep after reranking (up to 10 sources will be displayed)"
        )
        st.info("DeepSeek LLM metrics will be calculated for each response:\n- **Faithfulness**: Answer accuracy vs context\n- **Answer Relevancy**: How well answer addresses question")

        # Display reranker status
        display_reranker_status()
        st.markdown("---")

        # Load existing vector database section
        st.markdown('<div class="section-title">Load Existing Database</div>', unsafe_allow_html=True)
        metadata = load_vector_db_metadata()
        if metadata:
            db_options = []
            for db_name, info in metadata.items():
                try:
                    created_date = datetime.fromisoformat(info['created_at']).strftime("%Y-%m-%d %H:%M")
                    files_info = f"{info.get('total_files', 0)} files"
                    db_options.append(f"{db_name} ({created_date}, {files_info})")
                except Exception as e:
                    db_options.append(f"{db_name} (Invalid metadata)")

            selected_db = st.selectbox("Select a saved vector database:", ["None"] + db_options)

            if st.button("Load Selected Database") and selected_db != "None":
                db_name = selected_db.split(" (")[0]
                with st.spinner("Loading vector database..."):
                    try:
                        embeddings = AzureOpenAIEmbeddings(
                            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
                            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                        )
                        vectorstore, document_sources, uploaded_files_data = load_vector_db(db_name, embeddings)
                        if vectorstore:
                            conversation = get_conversation_chain_with_sources(vectorstore, faithfulness_threshold, reranker_top_n)
                            if conversation:
                                st.session_state.conversation = conversation
                                st.session_state.document_sources = document_sources or {}
                                st.session_state.current_db_name = db_name
                                st.session_state.processing_stats = metadata[db_name].get('stats', {})
                                st.session_state.uploaded_file_bytes = uploaded_files_data or {}
                                success_msg = f"Successfully loaded database: {db_name}"
                                if uploaded_files_data:
                                    success_msg += f" (PDF viewing available for {len(uploaded_files_data)} files)"
                                else:
                                    success_msg += " (PDF viewing not available)"
                                st.success(success_msg)
                            else:
                                st.error("Failed to create conversation chain")
                        else:
                            st.error("Failed to load database")
                    except Exception as e:
                        st.error(f"Error loading database: {e}")
        else:
            st.info("No saved vector databases found.")

        st.markdown("---")

        # Document upload section
        st.markdown('<div class="section-title">Document Upload</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF files or images",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        db_name = st.text_input("Database Name (for saving):", value=f"db_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if st.button("Process Documents"):
            if uploaded_files and db_name:
                with st.spinner("Processing documents..."):
                    try:
                        uploaded_files_data = {file.name: file.read() for file in uploaded_files}
                        for file in uploaded_files:
                            file.seek(0)
                        st.session_state.uploaded_file_bytes = uploaded_files_data

                        client = AzureOpenAI(
                            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                        )

                        documents, stats, document_sources = get_documents_text_with_sources(uploaded_files, client)
                        if documents:
                            st.session_state.processing_stats = stats
                            st.session_state.document_sources = document_sources
                            chunked_documents = get_text_chunks_with_sources(documents)
                            vectorstore = get_vectorstore_from_documents(chunked_documents)

                            if vectorstore:
                                file_names = [file.name for file in uploaded_files]
                                if save_vector_db(vectorstore, db_name, document_sources, uploaded_files_data):
                                    save_vector_db_metadata(db_name, file_names, stats)
                                    conversation = get_conversation_chain_with_sources(vectorstore, faithfulness_threshold, reranker_top_n)
                                    if conversation:
                                        st.session_state.conversation = conversation
                                        st.session_state.current_db_name = db_name
                                        st.success(f"Documents processed and saved as '{db_name}'!")
                                    else:
                                        st.error("Failed to create conversation chain.")
                                else:
                                    st.error("Failed to save vector database.")
                            else:
                                st.error("Failed to create vector database.")
                        else:
                            st.error("Could not extract any text from the uploaded files.")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
            else:
                st.warning("Please upload files and provide a database name.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.processing_stats:
            st.markdown("---")
            st.markdown('<div class="section-title">Processing Summary</div>', unsafe_allow_html=True)
            display_stats(st.session_state.processing_stats)
            if st.session_state.current_db_name:
                st.info(f"Current Database: {st.session_state.current_db_name}")

    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if not st.session_state.conversation:
        display_welcome()
    else:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if message.type == 'human' else "assistant"
            with st.chat_message(role):
                if role == "user":
                    st.markdown(message.content)
                else:
                    col1, col2 = st.columns([2, 3])
                    response_index = i // 2

                    with col1:
                        st.markdown(message.content)
                        if response_index < len(st.session_state.response_sources):
                            response_data = st.session_state.response_sources[response_index]
                            display_ragas_metrics(response_data.get('ragas_metrics'))

                    with col2:
                        # Wrap the content in a div to apply the CSS class for the vertical line
                        st.markdown('<div class="vertical-divider">', unsafe_allow_html=True)
                        if response_index < len(st.session_state.response_sources):
                            response_data = st.session_state.response_sources[response_index]
                            display_all_sources_interactive(response_data.get('sources', []), response_index)
                        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input and new response handling
        if user_question := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                col1, col2 = st.columns([2, 3])
                with col1:
                    with st.spinner("Searching and reranking documents..."):
                        answer, sources, ragas_metrics = handle_user_input_with_deepseek(user_question, faithfulness_threshold)
                        if answer:
                            st.markdown(answer)
                    display_ragas_metrics(ragas_metrics)

                with col2:
                    # Wrap the content in a div to apply the CSS class for the vertical line
                    st.markdown('<div class="vertical-divider">', unsafe_allow_html=True)
                    response_index = len(st.session_state.response_sources) - 1
                    display_all_sources_interactive(sources, response_index)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
