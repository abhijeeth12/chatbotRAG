from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import os
import uuid
import logging
import time
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sentence_transformers import SentenceTransformer
import io
import PyPDF2
import re
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'document_data.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Constants
OLLAMA_API = "http://localhost:11434/api"
ALLOWED_EXTENSIONS = {'pdf'}
SIMILARITY_THRESHOLD = 0.75
TOP_K = 3
HEADING_SIMILARITY_THRESHOLD = 0.65

# Debugging flag
DEBUG = True  # Debugging enabled by default

# Initialize components
document_chunks = {}
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

def log_debug(message):
    """Helper function for consistent debug logging"""
    if DEBUG:
        logger.debug(message)
        print(f"[DEBUG] {message}")

# Load existing data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        document_chunks = json.load(f)
    log_debug(f"Loaded {len(document_chunks)} documents from storage")

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_headings_and_content(pdf_url=None, raw_text=None, heading_pattern=r"^(#+)?\s*([A-Z][A-Za-z0-9 \-:]+)$"):
    """
    Enhanced heading extraction that handles:
    - Markdown-style headings (#, ##)
    - Normal text headings
    - More flexible formatting
    """
    chunks = {}
    current_heading = None
    current_content = []

    # Get text from either PDF or raw text
    text = ""
    if pdf_url:
        try:
            response = requests.get(pdf_url, timeout=20)
            pdf_stream = io.BytesIO(response.content)
            with pdfplumber.open(pdf_stream) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"PDF processing error: {e}")
            return chunks
    elif raw_text:
        text = raw_text
    else:
        return chunks

    # Debug original text
    print(f"\n[DEBUG] Raw text sample:\n{text[:500]}...\n")

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Enhanced heading detection
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            # Save previous heading content
            if current_heading:
                chunks[current_heading] = "\n".join(current_content).strip()
            
            # Get clean heading text (without markdown markers)
            current_heading = heading_match.group(2) if heading_match.group(2) else line
            current_content = []
            print(f"[DEBUG] Found heading: {current_heading}")  # Debug output
        elif current_heading:
            current_content.append(line)

    # Add the last heading
    if current_heading:
        chunks[current_heading] = "\n".join(current_content).strip()

    print(f"\n[DEBUG] Extracted {len(chunks)} headings")  # Debug count
    return chunks

def extract_text_from_pdf(filepath):
    """Extract text from PDF with error handling"""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def chunk_text(text, chunk_size=1000):
    """Create context-preserving chunks with headings"""
    heading_content = extract_headings_and_content(raw_text=text)
    chunks = []
    
    if not heading_content:
        log_debug("No headings found, using standard chunking")
        sections = re.split(r'\n\s*\n', text)
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += "\n\n" + section
            else:
                chunks.append(current_chunk.strip())
                current_chunk = section
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    for heading, content in heading_content.items():
        chunk = f"## {heading}\n{content}"
        while len(chunk) > chunk_size:
            split_pos = max(
                chunk.rfind('\n\n', 0, chunk_size),
                chunk.rfind('. ', 0, chunk_size)
            )
            if split_pos == -1:
                split_pos = chunk_size
            chunks.append(chunk[:split_pos].strip())
            chunk = f"## {heading}\n" + chunk[split_pos:].lstrip()
        chunks.append(chunk.strip())
    return chunks

def get_relevant_chunks(query, chunks):
    """Get most relevant chunks with similarity scores"""
    if not chunks:
        log_debug("⚠️ Warning: No chunks available for similarity calculation")
        return []
    
    # Generate embeddings
    query_embed = embedding_model.encode([query])
    chunk_embeds = embedding_model.encode(chunks)
    
    # Reshape arrays for sklearn compatibility
    if len(query_embed.shape) == 1:
        query_embed = query_embed.reshape(1, -1)
    if len(chunk_embeds.shape) == 1:
        chunk_embeds = chunk_embeds.reshape(1, -1)
    
    try:
        sims = cosine_similarity(query_embed, chunk_embeds)[0]
        top_indices = np.argsort(sims)[-TOP_K:][::-1]
        return [(chunks[i], sims[i]) for i in top_indices]
    except ValueError as e:
        log_debug(f"⚠️ Similarity calculation failed: {str(e)}")
        return []

def query_llama(prompt, model="llama3"):
    """Query LLM with debug logging"""
    try:
        log_debug(f"Sending to LLM: {prompt[:200]}...")
        start_time = time.time()
        
        response = requests.post(
            f"{OLLAMA_API}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json().get("response", "").strip()
        elapsed = time.time() - start_time
        log_debug(f"LLM response ({elapsed:.2f}s): {result[:200]}...")
        return result
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return "I encountered an error processing your request."

# Routes
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:  # Match the frontend key
        return jsonify({"success": False, "message": "No files uploaded"}), 400
    
    files = request.files.getlist('files[]')  # Get as list
    uploaded_docs = []
    
    for file in files:
        if not file or file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            continue
            
        try:
            file_id = str(uuid.uuid4())
            filename = f"{file_id}.pdf"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                continue
                
            # Process document
            text = extract_text_from_pdf(filepath)
            chunks = chunk_text(text) if text else []
            
            document_chunks[file_id] = {
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "chunks": chunks
            }
            
            uploaded_docs.append({
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "num_chunks": len(chunks)
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            continue
    
    if uploaded_docs:
        with open(DATA_FILE, 'w') as f:
            json.dump(document_chunks, f)
            
        return jsonify({
            "success": True,
            "message": f"Uploaded {len(uploaded_docs)} files",
            "documents": uploaded_docs
        })
    
    return jsonify({"success": False, "message": "No valid files processed"}), 400

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Enhanced RAG with hierarchical heading retrieval"""
    log_debug("Chat request received")
    
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON required"}), 400
    
    data = request.get_json()
    user_message = data.get('message', '').strip()
    document_ids = data.get('documents', [])
    
    if not user_message:
        return jsonify({"success": False, "message": "Message required"}), 400
    
    try:
        # 1. Retrieve document chunks
        all_chunks = []
        for doc_id in document_ids:
            if doc_id in document_chunks:
                all_chunks.extend(document_chunks[doc_id]['chunks'])
            else:
                log_debug(f"Document ID not found: {doc_id}")
        
        if not all_chunks:
            log_debug("No document chunks available - using general knowledge")
            response = query_llama(user_message)
            return jsonify({"success": True, "response": response})
        
        # 2. Extract and analyze headings
        combined_text = "\n".join(chunk for chunk in all_chunks)
        print(f"\n[DEBUG] Combined text sample:\n{combined_text[:1000]}...\n")  # Show sample
        
        heading_content = extract_headings_and_content(raw_text=combined_text)
        headings = list(heading_content.keys())
        log_debug(f"Extracted headings:")
        log_debug(f"Extracted headings:\n{'- ' + '\n- '.join(headings)}")
        
        # 3. Keyword extraction
        prompt = f"""Analyze this query and extract key terms that should appear in the answer.
Return ONLY a comma-separated list of 5-7 keywords/phrases.

Query: {user_message}"""
        
        keyword_string = query_llama(prompt)
        keywords = [k.strip() for k in keyword_string.split(",")] if keyword_string else []
        search_terms = f"{user_message} {' '.join(keywords)}"
        
        # 4. Heading-based retrieval (CRUCIAL PART)
        relevant_excerpts = []
        if headings:
            log_debug("Attempting heading-based retrieval...")
            heading_embeddings = embedding_model.encode(headings)
            search_embedding = embedding_model.encode([user_message])
            
            similarities = cosine_similarity(search_embedding, heading_embeddings)[0]
            heading_scores = sorted(zip(headings, similarities), 
                                  key=lambda x: x[1], reverse=True)
            
            # Get headings above threshold
            relevant_headings = [h for h, s in heading_scores 
                               if s > HEADING_SIMILARITY_THRESHOLD][:3]
            
            if relevant_headings:
                log_debug(f"Found relevant headings: {relevant_headings}")
                for heading in relevant_headings:
                    content = heading_content.get(heading, "")
                    if content:
                        relevant_excerpts.append(f"## {heading}\n{content[:500]}")
        
        # 5. Fallback to chunk-based retrieval if needed
        if not relevant_excerpts:
            log_debug("Using chunk-based retrieval fallback...")
            relevant_chunks = get_relevant_chunks(search_terms, all_chunks)
            relevant_excerpts = [f"### Excerpt\n{chunk[0][:500]}" for chunk in relevant_chunks]
        
        # 6. Generate response
        context = "\n\n".join(relevant_excerpts)[:5000]  # Limit context size
        prompt = f"""Answer the question based on these document sections:

Question: {user_message}

Document Context:
{context}

Provide a comprehensive answer that:
1. Directly addresses the question
2. References relevant sections
3. Synthesizes information when needed
4. Is accurate and concise
5.Remember that your response is directly displayed on my website related to RAG so provide response accordingly"""

        response = query_llama(prompt)
        return jsonify({
            "success": True,
            "response": response,
            "used_headings": relevant_headings if headings else [],
            "context": context
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "message": "Processing error",
            "response": "Sorry, I encountered an error"
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all stored documents"""
    documents = []
    for doc_id, doc_data in document_chunks.items():
        documents.append({
            "id": doc_id,
            "name": doc_data.get("name", "Unknown"),
            "size": doc_data.get("size", 0),
            "num_chunks": len(doc_data.get("chunks", []))
        })
    return jsonify({"success": True, "documents": documents})
@app.route('/',methods=['GET'])
def index():
    """Index route"""
    return render_template('index.html');
if __name__ == '__main__':
    logger.info("Starting application")
    app.run(host='0.0.0.0', port=5000, debug=True)