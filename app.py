from flask import Flask, request, jsonify, render_template, send_from_directory
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
import fitz  # PyMuPDF

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
DATA_FILE = 'document_data.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Constants
OLLAMA_API = "http://localhost:11434/api"
ALLOWED_EXTENSIONS = {'pdf'}
SIMILARITY_THRESHOLD = 0.75
TOP_K = 3
HEADING_SIMILARITY_THRESHOLD = 0.65

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
    logger.debug(message)
    print(f"[DEBUG] {message}")

# Load existing data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        document_chunks = json.load(f)
    # Convert stored embeddings back to NumPy arrays
    for doc in document_chunks.values():
        if "chunk_embeddings" in doc:
            doc["chunk_embeddings"] = [np.array(emb) for emb in doc["chunk_embeddings"]]
    log_debug(f"Loaded {len(document_chunks)} documents from storage")

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_images_from_pdf(filepath):
    """Extract images from PDF and associate with nearby headings using PyMuPDF"""
    images = []
    try:
        doc = fitz.open(filepath)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get text blocks to identify headings
            text_blocks = page.get_text("dict")["blocks"]
            headings = []
            for block in text_blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"]).strip()
                        # Simple heading detection
                        if text and re.match(r"^[A-Z][A-Za-z0-9 \-:]+$", text):
                            headings.append({"text": text, "y0": line["bbox"][1]})
            
            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join(app.config['IMAGE_FOLDER'], image_filename)
                
                # Save image to disk
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Find nearest heading
                image_rect = page.get_image_bbox(img)
                image_y = image_rect.y0
                nearest_heading = None
                min_distance = float('inf')
                for heading in headings:
                    distance = abs(heading["y0"] - image_y)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_heading = heading["text"]
                
                # Store image metadata
                images.append({
                    "filename": image_filename,
                    "heading": nearest_heading or "No heading",
                    "page": page_num + 1
                })
        
        doc.close()
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
    return images

def extract_text_from_pdf(filepath):
    """Extract text from PDF with error handling"""
    try:
        with pdfplumber.open(filepath) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def chunk_text(text, chunk_size=1000):
    """Create context-preserving chunks with headings"""
    chunks = []
    current_chunk = ""
    
    for paragraph in text.split('\n\n'):
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += '\n\n' + paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_relevant_chunks(query, chunks, chunk_embeddings):
    """Get most relevant chunks with similarity scores"""
    if not chunks or not chunk_embeddings:
        log_debug("No chunks or embeddings available")
        return []

    try:
        query_embed = embedding_model.encode([query])
        sims = cosine_similarity(query_embed, np.array(chunk_embeddings))[0]
        top_indices = np.argsort(sims)[-TOP_K:][::-1]
        return [(chunks[i], sims[i]) for i in top_indices]
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return []

def query_llama(prompt, model="llama3"):
    """Query LLM with debug logging"""
    try:
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
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return "I encountered an error processing your request."

# Routes
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({"success": False, "message": "No files uploaded"}), 400

    files = request.files.getlist('files[]')
    uploaded_docs = []
    errors = []

    for file in files:
        if not file or file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            errors.append(f"Invalid file type: {file.filename}")
            continue

        try:
            file_id = str(uuid.uuid4())
            filename = f"{file_id}.pdf"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process document
            text = extract_text_from_pdf(filepath)
            chunks = chunk_text(text) if text else []
            chunk_embeddings = embedding_model.encode(chunks).tolist() if chunks else []
            images = extract_images_from_pdf(filepath)

            document_chunks[file_id] = {
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "chunks": chunks,
                "chunk_embeddings": chunk_embeddings,
                "images": images,
                "upload_time": time.time()
            }

            uploaded_docs.append({
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "num_chunks": len(chunks),
                "num_images": len(images)
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            errors.append(f"Failed to process {file.filename}")

    if uploaded_docs:
        with open(DATA_FILE, 'w') as f:
            json.dump(document_chunks, f)

        response = {
            "success": True,
            "message": f"Processed {len(uploaded_docs)} files",
            "documents": uploaded_docs
        }
        if errors:
            response["warnings"] = errors
        return jsonify(response)

    return jsonify({
        "success": False,
        "message": "No valid files processed",
        "errors": errors
    }), 400

@app.route('/chat', methods=['POST'])
def handle_chat():
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON required"}), 400

    data = request.get_json()
    user_message = data.get('message', '').strip()
    document_ids = data.get('documents', [])

    if not user_message:
        return jsonify({"success": False, "message": "Message required"}), 400

    try:
        # Retrieve relevant content
        all_chunks = []
        all_embeddings = []
        all_images = []
        
        for doc_id in document_ids:
            if doc_id in document_chunks:
                doc = document_chunks[doc_id]
                all_chunks.extend(doc['chunks'])
                all_embeddings.extend(doc.get('chunk_embeddings', []))
                all_images.extend(doc.get('images', []))

        if not all_chunks:
            return jsonify({
                "success": True,
                "response": query_llama(user_message)
            })

        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(user_message, all_chunks, all_embeddings)
        context = "\n\n".join([chunk[0] for chunk in relevant_chunks])

        # Generate response
        prompt = f"""Answer based on this context:
{context}

Question: {user_message}
Provide a concise, accurate answer:"""
        
        response = query_llama(prompt)
        
        return jsonify({
            "success": True,
            "response": response,
            "context": context,
            "images": [img for img in all_images if img['page'] <= len(relevant_chunks)]
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "message": "Processing error",
            "error": str(e)
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    documents = [{
        "id": doc_id,
        "name": doc.get("name", "Unknown"),
        "size": doc.get("size", 0),
        "num_chunks": len(doc.get("chunks", [])),
        "num_images": len(doc.get("images", []))
    } for doc_id, doc in document_chunks.items()]
    
    return jsonify({"success": True, "documents": documents})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting application")
    app.run(host='0.0.0.0', port=5000, debug=True)