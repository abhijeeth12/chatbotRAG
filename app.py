from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
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
from pptx import Presentation
from docx import Document

# Initialize Flask application for web serving
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend compatibility

# Configuration settings
UPLOAD_FOLDER = 'Uploads'
IMAGE_FOLDER = 'Uploads/images'
DATA_FILE = 'document_data.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if it doesn't exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)   # Create image folder for extracted images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Constants for processing
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'docx', 'txt'}  # Supported file types
SIMILARITY_THRESHOLD = 0.75  # Threshold for chunk relevance
HEADING_SIMILARITY_THRESHOLD = 0.65  # Threshold for heading relevance
TOP_K = 3  # Number of top chunks to retrieve
CHUNK_SIZE = 1000  # Maximum size of each text chunk
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'  # xAI Grok API endpoint

# Debugging flag for enabling/disabling terminal output
DEBUG = True

# Initialize SentenceTransformer for embedding generation
print("Initializing SentenceTransformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized successfully")

# Initialize document storage
document_chunks = {}

# Helper function for consistent debug printing
def log_debug(message):
    """Print debug messages to terminal if DEBUG is True"""
    if DEBUG:
        print(f"[DEBUG] {message}")

# Load existing document data from JSON
def load_document_data():
    """Load document data from JSON file with error handling"""
    print("Loading document_data.json...")
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            for doc in data.values():
                if "chunk_embeddings" in doc:
                    doc["chunk_embeddings"] = [np.array(emb) for emb in doc["chunk_embeddings"]]
            print(f"Loaded {len(data)} documents from storage")
            return data
        else:
            print("document_data.json not found, initializing empty data")
            return {}
    except Exception as e:
        print(f"Error loading document_data.json: {str(e)}")
        return {}

# Save document data to JSON
def save_document_data(data):
    """Save document data to JSON file with error handling"""
    print("Saving document_data.json...")
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("document_data.json saved successfully")
    except Exception as e:
        print(f"Error saving document_data.json: {str(e)}")

# Initialize storage directories and load data
print("Creating upload and image folders if they don't exist...")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
print("Folders created successfully")
document_chunks = load_document_data()

# Validate file extensions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from PDF
def extract_text_from_pdf(filepath):
    """Extract text from PDF using PyPDF2 with error handling"""
    print("Extracting text from PDF...")
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        print("PDF text extraction successful")
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return ""

# Extract images from PDF
def extract_images_from_pdf(filepath):
    """Extract images from PDF using PyMuPDF with heading association"""
    images = []
    print("Extracting images from PDF...")
    try:
        doc = fitz.open(filepath)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]
            headings = []
            for block in text_blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"]).strip()
                        if text and re.match(r"^[A-Z][A-Za-z0-9 \-:]+$", text):
                            headings.append({"text": text, "y0": line["bbox"][1]})
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"pdf_image_{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join(app.config['IMAGE_FOLDER'], image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_rect = page.get_image_bbox(img)
                image_y = image_rect.y0
                nearest_heading = None
                min_distance = float('inf')
                for heading in headings:
                    distance = abs(heading["y0"] - image_y)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_heading = heading["text"]
                images.append({
                    "filename": image_filename,
                    "heading": nearest_heading or f"Image from page {page_num + 1}",
                    "page": page_num + 1
                })
        doc.close()
        print(f"Extracted {len(images)} images from PDF")
    except Exception as e:
        print(f"Error extracting PDF images: {str(e)}")
    return images

# Extract text and headings from PPTX
def extract_text_from_pptx(filepath):
    """Extract text and headings from PPTX using python-pptx"""
    text = ""
    headings = []
    print("Extracting text and headings from PPTX...")
    try:
        prs = Presentation(filepath)
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
                    if shape.is_placeholder and shape.placeholder_format.type in (1, 2):
                        headings.append({"text": shape.text, "slide": slide_num})
            text += slide_text + "\n"
        print(f"PPTX headings retrieved: {len(headings)} headings")
    except Exception as e:
        print(f"Error extracting PPTX text/headings: {str(e)}")
        print("No headings found for PPTX")
    return text, headings

# Extract images from PPTX
def extract_images_from_pptx(filepath):
    """Extract images from PPTX using python-pptx"""
    images = []
    print("Extracting images from PPTX...")
    try:
        prs = Presentation(filepath)
        for slide_num, slide in enumerate(prs.slides, 1):
            for shape in slide.shapes:
                if shape.shape_type == 13:
                    image = shape.image
                    image_bytes = image.blob
                    image_ext = image.ext
                    image_filename = f"pptx_image_{uuid.uuid4()}.{image_ext}"
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], image_filename)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    heading = f"Image from slide {slide_num}"
                    images.append({"filename": image_filename, "heading": heading, "slide": slide_num})
        print(f"Extracted {len(images)} images from PPTX")
    except Exception as e:
        print(f"Error extracting PPTX images: {str(e)}")
    return images

# Extract text and headings from DOCX
def extract_text_from_docx(filepath):
    """Extract text and headings from DOCX using python-docx"""
    text = ""
    headings = []
    print("Extracting text and headings from DOCX...")
    try:
        doc = Document(filepath)
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                headings.append({"text": para.text, "level": para.style.name})
            text += para.text + "\n"
        print(f"DOCX headings retrieved: {len(headings)} headings")
    except Exception as e:
        print(f"Error extracting DOCX text/headings: {str(e)}")
        print("No headings found for DOCX")
    return text, headings

# Extract text from TXT
def extract_text_from_txt(filepath):
    """Extract text from TXT files"""
    text = ""
    headings = []
    print("Extracting text from TXT...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print("No headings found for TXT (plain text)")
    except Exception as e:
        print(f"Error extracting TXT text: {str(e)}")
    return text, headings

# Extract headings and content
def extract_headings_and_content(pdf_url=None, raw_text=None, filepath=None, heading_pattern=r"^(#+)?\s*([A-Z][A-Za-z0-9 \-:]+)$"):
    """Extract headings and content from raw text or file with image association"""
    chunks = {}
    images = []
    print("Extracting headings and content...")
    text = raw_text or ""
    if filepath:
        file_ext = filepath.rsplit('.', 1)[1].lower()
        if file_ext == 'pdf':
            try:
                with pdfplumber.open(filepath) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                images = extract_images_from_pdf(filepath)
            except Exception as e:
                print(f"PDF processing error: {str(e)}")
                return chunks, images
        elif file_ext == 'pptx':
            text, headings = extract_text_from_pptx(filepath)
            images = extract_images_from_pptx(filepath)
            chunks = {h["text"]: "" for h in headings}
            print(f"PPTX headings retrieved: {len(chunks)} headings")
            return chunks, images
        elif file_ext == 'docx':
            text, headings = extract_text_from_docx(filepath)
            chunks = {h["text"]: "" for h in headings}
            print(f"DOCX headings retrieved: {len(chunks)} headings")
            return chunks, images
        elif file_ext == 'txt':
            text, headings = extract_text_from_txt(filepath)
            chunks = {"Text": text} if text else {}
            print("No headings found for TXT")
            return chunks, images
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            current_heading = heading_match.group(2) or line
            chunks[current_heading] = ""
            print(f"Found heading: {current_heading}")
        elif chunks:
            last_heading = list(chunks.keys())[-1]
            chunks[last_heading] += line + "\n"
    print(f"Extracted {len(chunks)} headings, {len(images)} images")
    return chunks, images

# Chunk text into smaller segments
def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Create context-preserving chunks with headings"""
    print("Chunking text...")
    heading_content = extract_headings_and_content(raw_text=text)[0]
    chunks = []
    if not heading_content:
        print("No headings found, using standard chunking")
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
        print(f"Text split into {len(chunks)} chunks")
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
    print(f"Text split into {len(chunks)} chunks")
    return chunks

# Get relevant chunks based on query
def get_relevant_chunks(query, chunks, chunk_embeddings):
    """Retrieve most relevant chunks using cosine similarity"""
    print("Computing relevant chunks...")
    if not chunks or not chunk_embeddings:
        print("No chunks or embeddings available for similarity calculation")
        return []
    try:
        query_embed = embedding_model.encode([query])
        chunk_embeds = np.array(chunk_embeddings)
        if len(query_embed.shape) == 1:
            query_embed = query_embed.reshape(1, -1)
        if len(chunk_embeds.shape) == 1:
            chunk_embeds = chunk_embeds.reshape(1, -1)
        sims = cosine_similarity(query_embed, chunk_embeds)[0]
        top_indices = np.argsort(sims)[-TOP_K:][::-1]
        print(f"Found {len(top_indices)} relevant chunks")
        return [(chunks[i], sims[i]) for i in top_indices if sims[i] > SIMILARITY_THRESHOLD]
    except Exception as e:
        print(f"Similarity calculation failed: {str(e)}")
        return []

# Query Grok API
def query_grok(prompt):
    """Query xAI Grok API with error handling"""
    print("Sending prompt to Grok API...")
    try:
        response = requests.post(
            GROK_API_URL,
            headers={'Authorization': f'Bearer {os.environ.get("GROK_API_KEY")}'},
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3
            },
            timeout=60
        )
        response_data = response.json()
        if response.status_code != 200 or 'choices' not in response_data:
            print("Error from Grok API")
            raise Exception("Error from Grok API")
        result = response_data["choices"][0]["message"]["content"].strip()
        print("Received Grok API response")
        return result
    except Exception as e:
        print(f"Grok API query failed: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads for PDF, PPTX, DOCX, and TXT"""
    print("Received file upload request")
    if 'files[]' not in request.files:
        print("No files uploaded")
        return jsonify({"success": False, "message": "No files uploaded"}), 400
    files = request.files.getlist('files[]')
    uploaded_docs = []
    errors = []
    for file in files:
        if not file or file.filename == '':
            print("Empty file part detected")
            errors.append("Empty file part")
            continue
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            errors.append(f"Invalid file type: {file.filename}")
            continue
        try:
            file_id = str(uuid.uuid4())
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{file_id}.{file_ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Processing file: {file.filename}")
            file.save(filepath)
            if not os.path.exists(filepath):
                print(f"Failed to save: {file.filename}")
                errors.append(f"Failed to save: {file.filename}")
                continue
            text = ""
            chunks = []
            images = []
            headings = []
            if file_ext == 'pdf':
                text = extract_text_from_pdf(filepath)
                chunks_dict, images = extract_headings_and_content(filepath=filepath)
                chunks = chunk_text(text) if text else []
                headings = list(chunks_dict.keys())
            elif file_ext == 'pptx':
                text, headings_list = extract_text_from_pptx(filepath)
                images = extract_images_from_pptx(filepath)
                chunks = chunk_text(text) if text else []
                headings = [h["text"] for h in headings_list]
            elif file_ext == 'docx':
                text, headings_list = extract_text_from_docx(filepath)
                chunks = chunk_text(text) if text else []
                headings = [h["text"] for h in headings_list]
            elif file_ext == 'txt':
                text, headings_list = extract_text_from_txt(filepath)
                chunks = chunk_text(text) if text else []
                headings = [h["text"] for h in headings_list]
            chunk_embeddings = embedding_model.encode(chunks).tolist() if chunks else []
            document_chunks[file_id] = {
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "chunks": chunks,
                "chunk_embeddings": chunk_embeddings,
                "images": images,
                "upload_time": time.time(),
                "headings": headings
            }
            uploaded_docs.append({
                "id": file_id,
                "name": file.filename,
                "size": os.path.getsize(filepath),
                "num_chunks": len(chunks),
                "images": images
            })
            print(f"File {file.filename} processed successfully")
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            errors.append(f"Failed to process {file.filename}: {str(e)}")
    if uploaded_docs:
        save_document_data(document_chunks)
        response = {
            "success": True,
            "message": f"Processed {len(uploaded_docs)} files successfully",
            "documents": uploaded_docs
        }
        if errors:
            response["warnings"] = errors
        print(f"Uploaded {len(uploaded_docs)} documents")
        return jsonify(response)
    print("No valid files processed")
    return jsonify({"success": False, "message": "No valid files processed", "errors": errors}), 400

# Chat endpoint
@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat queries with document context"""
    print("Chat request received")
    if not request.is_json:
        print("JSON required for chat request")
        return jsonify({"success": False, "message": "JSON required"}), 400
    data = request.get_json()
    user_message = data.get('message', '').strip()
    document_ids = data.get('documents', [])
    if not user_message:
        print("Message required for chat")
        return jsonify({"success": False, "message": "Message required"}), 400
    try:
        all_chunks = []
        all_chunk_embeddings = []
        all_images = []
        for doc_id in document_ids:
            if doc_id in document_chunks:
                all_chunks.extend(document_chunks[doc_id]['chunks'])
                all_chunk_embeddings.extend(document_chunks[doc_id].get('chunk_embeddings', []))
                all_images.extend(document_chunks[doc_id].get('images', []))
            else:
                print(f"Document ID not found: {doc_id}")
        if not all_chunks:
            print("No document chunks available - using general knowledge")
            response = query_grok(user_message)
            return jsonify({"success": True, "response": response})
        print("Generating query embedding...")
        search_embedding = embedding_model.encode([user_message])
        print("Query embedding generated")
        combined_text = "\n".join(chunk for chunk in all_chunks)
        print(f"Combined text sample:\n{combined_text[:1000]}...")
        heading_content = extract_headings_and_content(raw_text=combined_text)[0]
        headings = list(heading_content.keys())
        print(f"Extracted headings:\n{'- ' + '\n- '.join(headings)}")
        prompt = f"""Analyze this query and extract key terms that should appear in the answer.
Return ONLY a comma-separated list of 5-7 keywords/phrases.

Query: {user_message}"""
        keyword_string = query_grok(prompt)
        keywords = [k.strip() for k in keyword_string.split(",")] if keyword_string else []
        search_terms = f"{user_message} {' '.join(keywords)}"
        relevant_excerpts = []
        relevant_images = []
        relevant_headings = []
        if headings:
            print("Attempting heading-based retrieval...")
            heading_embeddings = embedding_model.encode(headings)
            similarities = cosine_similarity(search_embedding, heading_embeddings)[0]
            heading_scores = sorted(zip(headings, similarities), key=lambda x: x[1], reverse=True)
            relevant_headings = [h for h, s in heading_scores if s > HEADING_SIMILARITY_THRESHOLD][:3]
            if relevant_headings:
                print(f"Found relevant headings: {relevant_headings}")
                for heading in relevant_headings:
                    content = heading_content.get(heading, "")
                    if content:
                        relevant_excerpts.append(f"## {heading}\n{content[:500]}")
                    for img in all_images:
                        if not all(key in img for key in ['filename', 'heading']):
                            print(f"Skipping invalid image: {img}")
                            continue
                        if img["heading"] == heading:
                            try:
                                similarity = similarities[headings.index(heading)] if heading in headings else 0
                                relevant_images.append({
                                    "url": f"/images/{img['filename']}",
                                    "heading": img["heading"],
                                    "similarity": float(similarity)
                                })
                            except ValueError as e:
                                print(f"Error finding heading index for {heading}: {e}")
                                relevant_images.append({
                                    "url": f"/images/{img['filename']}",
                                    "heading": img["heading"],
                                    "similarity": 0.0
                                })
        if not relevant_excerpts:
            print("Using chunk-based retrieval fallback...")
            relevant_chunks = get_relevant_chunks(search_terms, all_chunks, all_chunk_embeddings)
            relevant_excerpts = [f"### Excerpt\n{chunk[0][:500]}" for chunk in relevant_chunks]
        context = "\n\n".join(relevant_excerpts)[:5000]
        prompt = f"""Answer the question based on these document sections:

Question: {user_message}

Document Context:
{context}

Provide a comprehensive answer that:
1. Directly addresses the question
2. References relevant sections
3. Synthesizes information when needed
4. Is accurate and concise
5. Is suitable for a RAG-based website"""
        print(f"Sending prompt to Grok API with {len(relevant_excerpts)} excerpts and {len(relevant_images)} images")
        response = query_grok(prompt)
        print(f"Returning response with {len(relevant_images)} images")
        return jsonify({
            "success": True,
            "response": response,
            "used_headings": relevant_headings,
            "context": context,
            "images": relevant_images
        })
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Processing error",
            "response": f"Sorry, I encountered an error: {str(e)}"
        }), 500

# List documents endpoint
@app.route('/documents', methods=['GET'])
def list_documents():
    """Return list of uploaded documents"""
    print("Fetching document list")
    documents = []
    for doc_id, doc_data in document_chunks.items():
        documents.append({
            "id": doc_id,
            "name": doc_data.get("name", "Unknown"),
            "size": doc_data.get("size", 0),
            "num_chunks": len(doc_data.get("chunks", [])),
            "images": doc_data.get("images", [])
        })
    return jsonify({"success": True, "documents": documents})

# Serve images endpoint
@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the image folder"""
    print(f"Serving image: {filename}")
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# Index endpoint
@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML page"""
    print("Serving index.html")
    return render_template('index.html')

# Main entry point
if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
