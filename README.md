# PDF Document Assistant with Reranking

A sophisticated Streamlit application for analyzing PDF documents and images with AI-powered document reranking and quality evaluation.

## Features

### 🎯 Core Functionality
- **Multi-Format Support**: Process PDFs and image files (PNG, JPG, JPEG) simultaneously
- **Smart Reranking**: DeepSeek LLM reorders search results by relevance to provide the most accurate answers first
- **AI-Powered Chat**: Ask questions about your documents and get intelligent responses
- **Quality Metrics**: DeepSeek LLM evaluation for response quality (faithfulness and relevance scoring)

### 🔍 Advanced Capabilities
- **OCR Integration**: Automatic text extraction from images and low-text PDF pages using Azure Vision
- **Vector Database Management**: Save and load processed document databases for future use
- **Interactive PDF Viewer**: View source documents with page navigation controls
- **Source Attribution**: See exactly which documents and pages provided the information

### 📊 Quality Assurance
- **Faithfulness Scoring**: Measures how well the answer aligns with the source context (0.0-1.0 scale)
- **Answer Relevancy**: Evaluates how well the answer addresses the specific question
- **Configurable Thresholds**: Set minimum quality thresholds for answer acceptance

## Prerequisites

### API Keys Required
1. **Azure OpenAI** - For embeddings, chat, and vision capabilities
2. **DeepSeek API** - For document reranking and quality evaluation

### Python Dependencies
- Python 3.8+
- Streamlit
- Azure OpenAI SDK
- PyMuPDF (fitz)
- LangChain
- FAISS
- Requests
- Other standard data science libraries

## Installation

1. Clone or download the application files
2. Install required dependencies:

```bash
pip install streamlit openai langchain langchain-openai langchain-community faiss-cpu pymupdf requests python-dotenv
```

3. Configure your API keys in the `AZURE_CONFIG` and `DEEPSEEK_CONFIG` sections of `app_v3.py`

## Configuration

### Azure OpenAI Setup
Update the `AZURE_CONFIG` dictionary with your Azure OpenAI credentials:

```python
AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": "your-azure-openai-endpoint",
    "AZURE_OPENAI_API_KEY": "your-azure-openai-api-key",
    "OPENAI_API_VERSION": "2025-01-01-preview",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "your-chat-deployment",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": "your-embeddings-deployment"
}
```

### DeepSeek Setup
Update the `DEEPSEEK_CONFIG` dictionary with your DeepSeek API credentials:

```python
DEEPSEEK_CONFIG = {
    "DEEPSEEK_API_KEY": "your-deepseek-api-key",
    "DEEPSEEK_API_URL": "https://api.deepseek.com/v1/chat/completions"
}
```

## Usage

### Starting the Application
```bash
streamlit run app_v3.py
```

### Basic Workflow

1. **Upload Documents**: Select PDF files or images through the sidebar
2. **Process Documents**: Click "Process Documents" to create embeddings and vector database
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **View Results**: See answers with quality metrics and source attribution

### Advanced Features

#### Document Reranking
The application uses DeepSeek LLM to:
- Retrieve multiple documents (configurable up to 15)
- Rerank them by relevance to the question
- Display top N most relevant sources (configurable 1-10)

#### Quality Evaluation
Each response is evaluated on:
- **Faithfulness**: How accurately the answer reflects the source content
- **Answer Relevancy**: How well the answer addresses the specific question

#### Database Management
- Save processed documents as vector databases for future use
- Load previously created databases from the sidebar
- View processing statistics and file information

## Technical Architecture

### Components

1. **Document Processing Pipeline**
   - PDF text extraction using PyMuPDF
   - OCR for images and low-text pages using Azure Vision
   - Text chunking and embedding creation
   - Vector database storage with FAISS

2. **Retrieval and Reranking**
   - Initial document retrieval using similarity search
   - DeepSeek LLM-based relevance scoring and reranking
   - Contextual compression for optimal results

3. **Response Generation**
   - Azure OpenAI chat completion for answer generation
   - Conversation memory for contextual understanding
   - Source document attribution

4. **Quality Evaluation**
   - DeepSeek LLM-based metric calculation
   - Caching system for repeated evaluations
   - Configurable quality thresholds

### File Structure

- `app_v3.py` - Main application file
- `vector_databases/` - Directory for storing vector databases
- `ragas_cache/` - Directory for caching evaluation metrics
- `db_metadata.json` - Metadata about saved databases

## Customization

### Configuration Options

In the sidebar, you can adjust:
- **Faithfulness Threshold**: Minimum score required to provide an answer (0.0-1.0)
- **Reranker Top N**: Number of top documents to display after reranking (1-10)

### Modifying the Reranker

The `DeepSeekReranker` class can be customized for:
- Different top N values
- Alternative scoring mechanisms
- Fallback strategies

### Adding New File Types

Extend the `get_documents_text_with_sources` function to support additional file formats.

## Troubleshooting

### Common Issues

1. **API Configuration Errors**
   - Ensure all API keys are correctly configured
   - Verify endpoint URLs are accessible

2. **Document Processing Failures**
   - Check file permissions and formats
   - Ensure sufficient API quotas

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Process fewer documents simultaneously

### Performance Tips

- Use smaller chunk sizes for better precision
- Process documents in batches for large collections
- Enable caching for repeated evaluations

## License

This application is provided for educational and research purposes. Ensure you comply with the terms of service for all integrated APIs (Azure OpenAI, DeepSeek).

## Support

For issues related to:
- API configuration: Consult respective provider documentation
- Application functionality: Review the code comments and configuration options
- Performance optimization: Consider document preprocessing and chunking strategies

## Version History

- **v3**: Added DeepSeek reranking, enhanced quality metrics, improved UI
- **v2**: Added Azure Vision OCR, improved error handling
- **v1**: Basic PDF processing and chat functionality

---

*Note: This application processes sensitive documents. Ensure you have appropriate permissions and comply with data privacy regulations when using this tool.*
