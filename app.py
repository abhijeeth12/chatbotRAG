from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import json
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import fitz  # PyMuPDF
from pptx import Presentation
from docx import Document
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'docx', 'txt'}

# Initialize SentenceTransformer model
print("Initializing SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized successfully")

# Load or initialize document data
DOCUMENT_DATA_FILE = 'document_data.json'

def load_document_data():
    print("Loading document_data.json...")
    try:
        with open(DOCUMENT_DATA_FILE, 'r') as f:
            data = json.load(f)
        print("document_data.json loaded successfully")
        return data
    except FileNotFoundError:
        print("document_data.json not found, initializing empty data")
        return {"documents": []}

def save_document_data(data):
    print("Saving document_data.json...")
    with open(DOCUMENT_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print("document_data.json saved successfully")

document_data = load_document_data()

# Ensure upload and image folders exist
print("Creating upload and image folders if they don't exist...")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
print("Folders created successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    headings = []
    print("Extracting text and headings from PDF...")
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                words = page.extract_words()
                for word in words:
                    if word['text'].isupper() or (word['text'][0].isupper() and len(word['text']) > 3):
                        headings.append({"text": word['text'], "page": page_num})
        print(f"PDF headings retrieved: {len(headings)} headings")
    except Exception as e:
        print(f"Error extracting PDF text/headings: {str(e)}")
        print("No headings found for PDF")
    return text, headings

def extract_images_from_pdf(file_path):
    images = []
    print("Extracting images from PDF...")
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"pdf_image_{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join(app.config['IMAGE_FOLDER'], image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                heading = f"Image from page {page_num + 1}"
                images.append({"filename": image_filename, "heading": heading, "page": page_num + 1})
        print(f"Extracted {len(images)} images from PDF")
    except Exception as e:
        print(f"Error extracting PDF images: {str(e)}")
    return images

def extract_text_from_pptx(file_path):
    text = ""
    headings = []
    print("Extracting text and headings from PPTX...")
    try:
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
                    if shape.is_placeholder and shape.placeholder_format.type in (1, 2):  # Title or Subtitle
                        headings.append({"text": shape.text, "slide": slide_num})
            text += slide_text + "\n"
        print(f"PPTX headings retrieved: {len(headings)} headings")
    except Exception as e:
        print(f"Error extracting PPTX text/headings: {str(e)}")
        print("No headings found for PPTX")
    return text, headings

def extract_images_from_pptx(file_path):
    images = []
    print("Extracting images from PPTX...")
    try:
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides, 1):
            for shape in slide.shapes:
                if shape.shape_type == 13:  # Picture
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

def extract_text_from_docx(file_path):
    text = ""
    headings = []
    print("Extracting text and headings from DOCX...")
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                headings.append({"text": para.text, "level": para.style.name})
            text += para.text + "\n"
        print(f"DOCX headings retrieved: {len(headings)} headings")
    except Exception as e:
        print(f"Error extracting DOCX text/headings: {str(e)}")
        print("No headings found for DOCX")
    return text, headings

def extract_text_from_txt(file_path):
    text = ""
    headings = []
    print("Extracting text from TXT...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("No headings found for TXT (plain text)")  # TXT has no heading structure
    except Exception as e:
        print(f"Error extracting TXT text: {str(e)}")
    return text, headings

def chunk_text(text, max_chunk_size=500):
    print("Chunking text...")
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    print(f"Text split into {len(chunks)} chunks")
    return chunks

def generate_embeddings(chunks):
    print("Generating embeddings for chunks...")
    try:
        embeddings = model.encode(chunks, convert_to_tensor=False)
        print("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return []

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received file upload request")
    if 'files[]' not in request.files:
        print("No files uploaded")
        return jsonify({"success": False, "message": "No files uploaded"}), 400
    files = request.files.getlist('files[]')
    uploaded_documents = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_ext = filename.rsplit('.', 1)[1].lower()
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{file_ext}")
            print(f"Processing file: {filename}")
            file.save(file_path)
            file_size = os.path.getsize(file_path)
            text, headings, images = "", [], []
            if file_ext == 'pdf':
                text, headings = extract_text_from_pdf(file_path)
                images = extract_images_from_pdf(file_path)
            elif file_ext == 'pptx':
                text, headings = extract_text_from_pptx(file_path)
                images = extract_images_from_pptx(file_path)
            elif file_ext == 'docx':
                text, headings = extract_text_from_docx(file_path)
            elif file_ext == 'txt':
                text, headings = extract_text_from_txt(file_path)
            chunks = chunk_text(text)
            embeddings = generate_embeddings(chunks)
            document = {
                "id": file_id,
                "name": filename,
                "size": file_size,
                "chunks": chunks,
                "embeddings": embeddings.tolist(),
                "headings": headings,
                "images": images
            }
            document_data["documents"].append(document)
            uploaded_documents.append({"id": file_id, "name": filename, "size": file_size, "images": images})
            print(f"File {filename} processed successfully")
    save_document_data(document_data)
    print(f"Uploaded {len(uploaded_documents)} documents")
    return jsonify({"success": True, "documents": uploaded_documents})

@app.route('/chat', methods=['POST'])
def chat():
    print("Received chat query")
    data = request.get_json()
    if not data or 'message' not in data:
        print("No message provided in chat query")
        return jsonify({"success": False, "message": "No message provided"}), 400
    message = data['message']
    doc_ids = data.get('documents', [])
    relevant_chunks = []
    relevant_images = []
    print(f"Processing query: {message}")
    if doc_ids:
        print("Generating query embedding...")
        query_embedding = model.encode([message], convert_to_tensor=False)[0]
        print("Query embedding generated")
        for doc in document_data["documents"]:
            if doc["id"] in doc_ids:
                print(f"Searching document: {doc['name']}")
                embeddings = np.array(doc["embeddings"])
                similarities = cosine_similarity([query_embedding], embeddings)[0]
                top_indices = np.argsort(similarities)[-3:][::-1]
                print(f"Found {len(top_indices)} relevant chunks")
                for idx in top_indices:
                    if similarities[idx] > 0.2:
                        relevant_chunks.append(doc["chunks"][idx])
                        for image in doc["images"]:
                            if ("page" in image and image["page"] == idx + 1) or ("slide" in image and image["slide"] == idx + 1):
                                relevant_images.append({"url": f"/images/{image['filename']}", "heading": image["heading"], "similarity": similarities[idx]})
    context = "\n".join(relevant_chunks) if relevant_chunks else "No relevant document content found."
    prompt = f"Context: {context}\n\nQuestion: {message}\nAnswer:"
    print(f"Sending prompt to LLM with {len(relevant_chunks)} chunks and {len(relevant_images)} images")
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        response_data = response.json()
        if response.status_code != 200 or 'response' not in response_data:
            print("Error from LLM API")
            raise Exception("Error from LLM API")
        print("Received LLM response")
        return jsonify({"success": True, "response": response_data["response"], "images": relevant_images})
    except Exception as e:
        print(f"Chat query failed: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    print("Fetching document list")
    return jsonify({"documents": [{"id": doc["id"], "name": doc["name"], "size": doc["size"]} for doc in document_data["documents"]]})

@app.route('/images/<filename>')
def serve_image(filename):
    print(f"Serving image: {filename}")
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/', methods=['GET'])
def index():
    print("Serving index.html")
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
