const appState = {
    uploadedDocuments: [],
    chatHistory: [],
    currentProcessing: false
};

// Constants
const ALLOWED_EXTENSIONS = new Set(['pdf', 'pptx', 'docx', 'txt']);
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
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Element ${key} not found`);
        }
    }
}

function setupEventListeners() {
    if (elements.dropZone && elements.fileInput) {
        elements.dropZone.addEventListener('click', () => elements.fileInput.click());
        elements.fileInput.addEventListener('change', () => {
            if (elements.fileInput.files?.length > 0) {
                handleFileSelection(elements.fileInput.files);
            }
        });
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
    if (elements.clearDocsBtn) {
        elements.clearDocsBtn.addEventListener('click', clearAllDocuments);
    }
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
    if (!file || typeof file !== 'object' || !file.name || !file.size || !file.type) return false;
    if (file.size > MAX_FILE_SIZE) {
        showMessage('error', `File ${file.name} exceeds size limit (16MB)`);
        return false;
    }
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    if (!ALLOWED_EXTENSIONS.has(extension)) {
        showMessage('error', `File ${file.name} is not a supported format (PDF, PPTX, DOCX, TXT)`);
        return false;
    }
    return true;
}

async function handleFileSelection(files) {
    if (!files || files.length === 0) {
        showMessage('error', 'No files selected');
        return;
    }
    const validFiles = Array.from(files).filter(isValidFile);
    if (validFiles.length === 0) {
        showMessage('error', 'Please select supported files only (PDF, PPTX, DOCX, TXT)');
        return;
    }
    showLoading(true);
    const formData = new FormData();
    validFiles.forEach(file => formData.append('files[]', file));
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || `Upload failed: ${response.status}`);
        }
        if (data.success) {
            appState.uploadedDocuments = [...appState.uploadedDocuments, ...data.documents];
            updateDocumentList();
            updateFileCount();
            const imageCount = data.documents.reduce((sum, doc) => sum + (doc.images?.length || 0), 0);
            showMessage('success', `Files uploaded successfully! ${imageCount} images extracted.`);
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
    elements.documentList.innerHTML = appState.uploadedDocuments.map(doc => {
        // Group images by heading
        const imagesByHeading = {};
        if (doc.images && doc.images.length > 0) {
            doc.images.forEach(img => {
                const heading = img.heading || 'No heading';
                if (!imagesByHeading[heading]) {
                    imagesByHeading[heading] = [];
                }
                imagesByHeading[heading].push(img);
            });
        }
        const hasImages = Object.keys(imagesByHeading).length > 0;
        
        return `
            <div class="document-item" data-id="${doc.id}">
                <div class="document-info">
                    <div class="document-name">${doc.name}</div>
                    <div class="document-size">${formatFileSize(doc.size)}</div>
                    <div class="document-images-toggle">
                        <button class="view-images-btn" aria-expanded="false" aria-controls="images-${doc.id}" ${!hasImages ? 'disabled' : ''}>
                            ${hasImages ? `View Images (${doc.images.length})` : 'No Images Available'}
                        </button>
                        <div class="document-images" id="images-${doc.id}" style="display: none;">
                            ${hasImages ? Object.entries(imagesByHeading).map(([heading, imgs]) => `
                                <div class="images-section">
                                    <h4 class="images-heading">${heading}</h4>
                                    <div class="images-list">
                                        ${imgs.map(img => `
                                            <div class="image-preview">
                                                <img src="/images/${img.filename}" alt="Image for ${img.heading}" title="${img.heading}" loading="lazy" onerror="this.src='/fallback-image.png';" />
                                                <div class="image-description">${img.page ? `Page: ${img.page}` : img.slide ? `Slide: ${img.slide}` : ''}</div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `).join('') : '<p>No images extracted</p>'}
                        </div>
                    </div>
                </div>
                <i class="fas fa-times remove-doc" title="Remove document"></i>
            </div>
        `;
    }).join('');

    // Add event listeners for toggle buttons
    document.querySelectorAll('.view-images-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const imagesSection = e.target.closest('.document-images-toggle').querySelector('.document-images');
            const isExpanded = e.target.getAttribute('aria-expanded') === 'true';
            imagesSection.style.display = isExpanded ? 'none' : 'block';
            e.target.setAttribute('aria-expanded', !isExpanded);
            e.target.textContent = isExpanded ? `View Images (${imagesSection.querySelectorAll('.image-preview').length})` : 'Hide Images';
        });
    });

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
        const result = await response.json();
        if (!response.ok) throw new Error(result.message || `Server responded with status ${response.status}`);
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
            throw new Error(result.response || 'Failed to get response');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeThinkingMessage(thinkingMessage);
        addChatMessage('bot', `Sorry, I encountered an error: ${error.message}`);
        showMessage('error', error.message);
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
                    <img src="${img.url}" alt="Image for ${img.heading}" title="${img.heading}" loading="lazy" onerror="this.src='/fallback-image.png';" />
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