// State management
const appState = {
    uploadedDocuments: [],
    chatHistory: [],
    currentProcessing: false
};

// Constants
const ALLOWED_EXTENSIONS = new Set(['pdf']);
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB

// DOM elements
const elements = {
    fileInput: document.getElementById('fileInput'),
    dropZone: document.getElementById('dropZone'),
    documentList: document.getElementById('documentList'),
    fileCount: document.getElementById('fileCount'),
    clearDocsBtn: document.getElementById('clearDocsBtn'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    chatContainer: document.getElementById('chatContainer'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    fileIndicator: document.getElementById('fileIndicator')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    verifyElements();
    setupEventListeners();
    console.log('Application initialized');
});

function verifyElements() {
    // Verify all required elements exist
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Element ${key} not found`);
        }
    }
}

function setupEventListeners() {
    // File input via click or drop
    if (elements.dropZone && elements.fileInput) {
        elements.dropZone.addEventListener('click', () => elements.fileInput.click());

        // File selection
        elements.fileInput.addEventListener('change', () => {
            if (elements.fileInput.files?.length > 0) {
                handleFileSelection(elements.fileInput.files);
            }
        });

        // Drag and drop
        elements.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.dropZone.classList.add('drag-active');
        });

        elements.dropZone.addEventListener('dragleave', () => {
            elements.dropZone.classList.remove('drag-active');
        });

        elements.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.dropZone.classList.remove('drag-active');
            if (e.dataTransfer?.files?.length > 0) {
                handleFileSelection(e.dataTransfer.files);
            }
        });
    }

    // Clear documents
    if (elements.clearDocsBtn) {
        elements.clearDocsBtn.addEventListener('click', clearAllDocuments);
    }

    // Chat input
    if (elements.messageInput && elements.sendButton) {
        elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        elements.sendButton.addEventListener('click', sendMessage);
    }
}

function isValidFile(file) {
    // Check if file exists and has required properties
    if (!file || typeof file !== 'object') return false;
    if (!file.name || !file.size || !file.type) return false;
    
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
        showMessage('error', `File ${file.name} exceeds size limit (16MB)`);
        return false;
    }
    
    // Check extension safely
    const fileName = file.name || '';
    const extension = fileName.split('.').pop() || '';
    if (!ALLOWED_EXTENSIONS.has(extension)) {
        showMessage('error', `File ${file.name} is not a PDF`);
        return false;
    }
    
    return true;
}

async function handleFileSelection(files) {
    if (!files || files.length === 0) {
        showMessage('error', 'No files selected');
        return;
    }

    // Filter valid PDF files
    const validFiles = Array.from(files).filter(file => {
        if (!file || !file.name) return false;
        
        // Safely check extension
        const fileName = file.name || '';
        const extension = fileName.split('.').pop();
        return extension === 'pdf';
    });

    if (validFiles.length === 0) {
        showMessage('error', 'Please select PDF files only');
        return;
    }

    showLoading(true);
    const formData = new FormData();
    
    // Use 'files[]' as key for multiple files
    validFiles.forEach(file => {
        formData.append('files[]', file);  // Note the [] for multiple files
    });

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData  // No Content-Type header for FormData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || `Upload failed: ${response.status}`);
        }

        if (data.success) {
            appState.uploadedDocuments = data.documents;
            updateDocumentList();
            updateFileCount();
            showMessage('success', 'Files uploaded successfully!');
        } else {
            throw new Error(data.message || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showMessage('error', error.message);
    } finally {
        showLoading(false);
        elements.fileInput.value = '';
    }
}

function updateDocumentList() {
    elements.documentList.innerHTML = appState.uploadedDocuments.map(doc => `
        <div class="document-item" data-id="${doc.id}">
            <div class="document-info">
                <div class="document-name">${doc.name}</div>
                <div class="document-size">${formatFileSize(doc.size)}</div>
                <div class="document-images">
                    ${doc.images && doc.images.length > 0 ? 
                        doc.images.map(img => `
                            <div class="image-preview">
                                <img src="/images/${img.filename}" alt="${img.heading}" title="${img.heading}" />
                                <div class="image-heading">${img.heading}</div>
                            </div>
                        `).join('') : 'No images extracted'}
                </div>
            </div>
            <i class="fas fa-times remove-doc" title="Remove document"></i>
        </div>
    `).join('');

    document.querySelectorAll('.remove-doc').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const docId = btn.closest('.document-item').dataset.id;
            removeDocument(docId);
        });
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

function removeDocument(docId) {
    appState.uploadedDocuments = appState.uploadedDocuments.filter(doc => doc.id !== docId);
    updateDocumentList();
    updateFileCount();
    showMessage('info', 'Document removed');
}

function clearAllDocuments() {
    appState.uploadedDocuments = [];
    updateDocumentList();
    updateFileCount();
    showMessage('info', 'All documents cleared');
}

function updateFileCount() {
    const count = appState.uploadedDocuments.length;
    elements.fileCount.textContent = count;
    if (elements.fileIndicator) {
        elements.fileIndicator.style.display = count > 0 ? 'flex' : 'none';
    }
}

async function sendMessage() {
    const messageText = elements.messageInput.value.trim();
    if (!messageText || appState.currentProcessing) return;

    addChatMessage('user', messageText);
    elements.messageInput.value = '';
    appState.currentProcessing = true;

    const thinkingMessage = addThinkingMessage();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: messageText,
                documents: appState.uploadedDocuments.map(doc => doc.id)
            })
        });

        if (!response.ok) throw new Error(`Server responded with status ${response.status}`);

        const result = await response.json();
        if (result.success) {
            removeThinkingMessage(thinkingMessage);
            addChatMessage('bot', result.response, result.images || []);
            appState.chatHistory.push({
                user: messageText,
                bot: result.response,
                images: result.images || [],
                timestamp: new Date().toISOString()
            });
        } else {
            throw new Error(result.message || 'Failed to get response');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeThinkingMessage(thinkingMessage);
        addChatMessage('bot', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        appState.currentProcessing = false;
        showLoading(false);
    }
}

function addThinkingMessage() {
    const existingThinking = document.querySelector('.thinking-message');
    if (existingThinking) existingThinking.remove();

    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message bot-message thinking-message';
    thinkingDiv.innerHTML = `
        <div class="thinking-text">Thinking<span class="thinking-dots"></span></div>
        <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    elements.chatContainer.appendChild(thinkingDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    return thinkingDiv;
}

function removeThinkingMessage(thinkingDiv) {
    if (thinkingDiv && thinkingDiv.parentNode) {
        thinkingDiv.remove();
    }
}

function addChatMessage(sender, text, images = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let imagesHtml = '';
    if (images.length > 0) {
        imagesHtml = '<div class="message-images">' + 
            images.map(img => `
                <div class="image-container">
                    <img src="${img.url}" alt="${img.heading}" title="${img.heading}" />
                    <div class="image-caption">Heading: ${img.heading} (Similarity: ${img.similarity.toFixed(2)})</div>
                </div>
            `).join('') +
            '</div>';
    }

    messageDiv.innerHTML = `
        <div>${formatMessageText(text)}</div>
        ${imagesHtml}
        <span class="message-time">${timestamp}</span>
    `;
    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;

    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();
}

function formatMessageText(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    elements.fileInput.disabled = show;
    elements.dropZone.classList.toggle('disabled', show);
}

function showMessage(type, text) {
    const message = document.createElement('div');
    message.className = `notification ${type}`;
    message.innerHTML = `<i class="fas ${getNotificationIcon(type)}"></i><span>${text}</span>`;
    document.body.appendChild(message);
    setTimeout(() => message.remove(), 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}
