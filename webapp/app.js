class ChatApplication {
    constructor() {
        this.socket = null;
        this.conversationHistory = [];
        this.pendingMessages = [];
        this.streamingMessage = null;
        this.uuid = crypto.randomUUID();
        this.elements = {};
        
        // Configuration
        this.config = {
            wsUrl: 'ws://127.0.0.1:8000/ws',
            storageKeys: {
                multiChat: 'tensoralix-multi-chat',
                welcomeShown: 'tensoralix-welcome-shown',
                currentChatId: 'currentChatId',
                chatStartTimestamp: 'chatStartTimestamp'
            },
            maxTitleLength: 30
        };
    }

    // Initialize the application
    init() {
        this.cacheElements();
        this.bindEvents();
        this.updateTimeDisplay();
        this.loadChatFromStorage();
        this.handlePageLoad();
    }

    // Cache DOM elements to avoid repeated queries
    cacheElements() {
        this.elements = {
            chatContainer: document.getElementById('chat-container'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            attachmentButton: document.getElementById('attachment-button'),
            attachmentMenu: document.getElementById('attachment-menu'),
            mobileMenuButton: document.getElementById('mobile-menu'),
            mobileSidebar: document.getElementById('mobile-sidebar'),
            closeMobileMenu: document.getElementById('close-mobile-menu'),
            newChatButtons: [
                document.getElementById('new-chat'),
                document.getElementById('new-chat-mobile')
            ].filter(Boolean),
            currentTimeSpan: document.getElementById('current-time'),
            sidebarContainers: document.querySelectorAll('.sidebar .space-y-1')
        };
    }

    // Bind all event listeners
    bindEvents() {
        // Message input events
        this.elements.messageInput?.addEventListener('input', this.handleInputResize.bind(this));
        this.elements.messageInput?.addEventListener('keydown', this.handleKeydown.bind(this));
        this.elements.sendButton?.addEventListener('click', this.sendMessage.bind(this));

        // UI events
        this.elements.attachmentButton?.addEventListener('click', this.toggleAttachmentMenu.bind(this));
        this.elements.mobileMenuButton?.addEventListener('click', this.openMobileMenu.bind(this));
        this.elements.closeMobileMenu?.addEventListener('click', this.closeMobileMenu.bind(this));
        
        // New chat buttons
        this.elements.newChatButtons.forEach(button => {
            button.addEventListener('click', this.createNewChat.bind(this));
        });

        // Global events
        document.addEventListener('click', this.handleGlobalClick.bind(this));
        document.addEventListener('click', this.handleCodeCopy.bind(this));
        document.addEventListener('click', this.handleCodeEdit.bind(this));
        window.addEventListener('load', this.handlePageLoad.bind(this));
    }

    // Update time display
    updateTimeDisplay() {
        if (this.elements.currentTimeSpan) {
            const now = new Date();
            this.elements.currentTimeSpan.textContent = now.toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
        }
    }

    // Handle input textarea resizing
    handleInputResize(event) {
        const input = event.target;
        input.style.height = 'auto';
        input.style.height = `${input.scrollHeight}px`;
    }

    // Handle keyboard events
    handleKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    // Toggle attachment menu
    toggleAttachmentMenu(event) {
        event.stopPropagation();
        this.elements.attachmentMenu?.classList.toggle('hidden');
    }

    // Mobile menu handlers
    openMobileMenu() {
        const sidebar = this.elements.mobileSidebar;
        if (sidebar) {
            sidebar.classList.remove('hidden');
            setTimeout(() => {
                sidebar.querySelector('div')?.classList.remove('-translate-x-full');
            }, 10);
        }
    }

    closeMobileMenu() {
        const sidebar = this.elements.mobileSidebar;
        if (sidebar) {
            sidebar.querySelector('div')?.classList.add('-translate-x-full');
            setTimeout(() => {
                sidebar.classList.add('hidden');
            }, 300);
        }
    }

    // Handle global clicks (close menus)
    handleGlobalClick() {
        this.elements.attachmentMenu?.classList.add('hidden');
    }

    // Create new chat
    createNewChat() {
        const newChatId = crypto.randomUUID();
        const timestamp = new Date().toISOString();
        
        sessionStorage.setItem(this.config.storageKeys.currentChatId, newChatId);
        sessionStorage.setItem(this.config.storageKeys.chatStartTimestamp, timestamp);

        this.conversationHistory = [];
        this.elements.chatContainer.innerHTML = '';
        this.disconnectWebSocket();
        this.displayWelcomeMessage();
    }

    // Send message
    sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.conversationHistory.push({ role: 'user', content: message });
        
        this.ensureChatSession();
        this.ensureWebSocketConnection();
        
        this.clearInput();
        this.sendToWebSocket();
        this.showTypingIndicator();
        this.saveChatToStorage();
    }

    // Ensure chat session exists
    ensureChatSession() {
        if (!sessionStorage.getItem(this.config.storageKeys.currentChatId)) {
            sessionStorage.setItem(this.config.storageKeys.currentChatId, crypto.randomUUID());
        }
        if (!sessionStorage.getItem(this.config.storageKeys.chatStartTimestamp)) {
            sessionStorage.setItem(this.config.storageKeys.chatStartTimestamp, new Date().toISOString());
        }
    }

    // Clear message input
    clearInput() {
        this.elements.messageInput.value = '';
        this.elements.messageInput.style.height = 'auto';
    }

    // Send message to WebSocket
    sendToWebSocket() {
        const payload = JSON.stringify({ 
            conv_id: this.uuid, 
            history: [...this.conversationHistory] 
        });
        
        if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(payload);
        } else {
            this.pendingMessages.push(payload);
        }
    }

    // Ensure WebSocket connection
    ensureWebSocketConnection() {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.connectWebSocket();
        }
    }

    // Connect WebSocket
    connectWebSocket() {
        this.socket = new WebSocket(`${this.config.wsUrl}/${this.uuid}`);
        this.socket.binaryType = "arraybuffer";

        this.socket.onopen = () => {
            this.pendingMessages.forEach(msg => this.socket.send(msg));
            this.pendingMessages = [];
        };

        this.socket.onmessage = this.handleWebSocketMessage.bind(this);
        this.socket.onclose = () => console.warn("WebSocket closed.");
        this.socket.onerror = (error) => console.error("WebSocket error:", error);
    }

    // Handle WebSocket messages
    handleWebSocketMessage(event) {
        this.removeTypingIndicator();

        try {
            const response = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(event.data)));
            
            switch (response.type) {
                case 'stream':
                    this.handleStreamMessage(response);
                    break;
                case 'end_stream':
                    this.handleEndStream();
                    break;
                case 'text':
                    this.handleTextMessage(response);
                    break;
                case 'image':
                    this.handleImageMessage(response);
                    break;
            }
        } catch (err) {
            console.error("Failed to parse response:", err);
        }
    }

    // Handle streaming message
    handleStreamMessage(response) {
        if (!this.streamingMessage) {
            this.initializeStreamingMessage();
        }
        
        this.streamingMessage.content += response.content;
        this.updateStreamingDisplay();
    }

    // Initialize streaming message display
    initializeStreamingMessage() {
        const uniqueId = `streaming-text-${Date.now()}`;
        this.streamingMessage = {
            content: '',
            element: this.createMessageElement('assistant', '', uniqueId),
            id: uniqueId
        };
        
        this.elements.chatContainer.appendChild(this.streamingMessage.element);
        this.scrollToBottom();
    }

    // Update streaming message display
    updateStreamingDisplay() {
        const safeHTML = this.parseMarkdown(this.streamingMessage.content);
        const contentElement = document.getElementById(this.streamingMessage.id);
        
        if (contentElement) {
            contentElement.innerHTML = safeHTML;
            this.addCodeCopyButtons(this.streamingMessage.element);
            this.fixStyle(this.streamingMessage.element);
        }
    }

    // Handle end of stream
    handleEndStream() {
        if (this.streamingMessage) {
            this.conversationHistory.push({ 
                role: 'assistant', 
                content: this.streamingMessage.content 
            });
            this.saveChatToStorage();
            this.streamingMessage = null;
        }
    }

    // Handle text message
    handleTextMessage(response) {
        this.conversationHistory.push({ role: 'assistant', content: response.content });
        this.addMessage(response.content, 'assistant');
    }

    // Handle image message
    handleImageMessage(response) {
        const dataUri = `data:image/png;base64,${response.content.trim()}`;
        const imageHtml = `<div class="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 mt-2"><img src="${dataUri}" alt="Generated Image" class="w-full h-auto"/></div>`;
        
        this.conversationHistory.push({ role: 'assistant', content: imageHtml });
        this.addMessage(imageHtml, 'assistant');
    }

    // Disconnect WebSocket
    disconnectWebSocket() {
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
            this.socket.close(1000, 'Client disconnecting');
        }
    }

    // Add message to chat
    addMessage(text, sender) {
        const messageElement = this.createMessageElement(sender, text);
        this.elements.chatContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    // Create message element
    createMessageElement(sender, text, uniqueId = null) {
        const messageDiv = document.createElement('div');
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const safeHTML = this.parseMarkdown(text);
        const contentId = uniqueId || '';

        messageDiv.className = `message flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;

        if (sender === 'user') {
            messageDiv.innerHTML = this.getUserMessageHTML(safeHTML, time);
        } else {
            messageDiv.innerHTML = this.getAssistantMessageHTML(safeHTML, time, contentId);
        }

        this.addCodeCopyButtons(messageDiv);
        return messageDiv;
    }

    // Get user message HTML
    getUserMessageHTML(content, time) {
        return `
            <div class="max-w-[85%] md:max-w-[75%]">
                <div class="user-bubble rounded-2xl rounded-tr-none p-5 shadow-sm message-bubble">
                    <div class="parsed-markdown">${content}</div>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-2 mr-1 text-right flex items-center justify-end">
                    <i class="fas fa-clock mr-1"></i>${time}
                </div>
            </div>
            <div class="flex-shrink-0 w-10 h-10 rounded-full user-avatar flex items-center justify-center text-white ml-3">
                <i class="fas fa-user"></i>
            </div>
        `;
    }

    // Get assistant message HTML
    getAssistantMessageHTML(content, time, contentId) {
        const idAttribute = contentId ? `id="${contentId}"` : '';
        return `
            <div class="flex-shrink-0 w-10 h-10 rounded-full ai-avatar flex items-center justify-center text-white mr-3">
                <i class="fas fa-robot"></i>
            </div>
            <div class="max-w-[85%] md:max-w-[75%]">
                <div class="assistant-bubble rounded-2xl rounded-tl-none p-5 shadow-sm message-bubble">
                    <div class="parsed-markdown text-gray-800 dark:text-gray-100" ${idAttribute}>${content}</div>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-2 ml-1 flex items-center">
                    <i class="fas fa-clock mr-1"></i>${time}
                </div>
            </div>
        `;
    }

    // Parse markdown safely
    parseMarkdown(text) {
        return DOMPurify.sanitize(marked.parse(text, {
            highlight: (code, lang) => {
                const validLang = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language: validLang }).value;
            }
        }));
    }

    // Add copy buttons to code blocks
    addCodeCopyButtons(element) {
        element.querySelectorAll('pre > code').forEach(codeBlock => {
            const pre = codeBlock.parentNode;
            
            // Check if already wrapped in container
            if (pre.parentNode.classList.contains('code-container')) {
                // Container exists, check if buttons exist
                if (!pre.parentNode.querySelector('.code-buttons')) {
                    const buttonsContainer = this.createCodeButtons();
                    pre.parentNode.appendChild(buttonsContainer);
                }
                return;
            }

            // Create new container with buttons
            const container = document.createElement('div');
            container.className = 'code-container relative';
            pre.parentNode.replaceChild(container, pre);
            container.appendChild(pre);

            // Add buttons container
            const buttonsContainer = this.createCodeButtons();
            container.appendChild(buttonsContainer);
        });
    }

    // Create the buttons container with Copy and Edit buttons
    createCodeButtons() {
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'code-buttons absolute top-2 right-2 flex gap-2';
        
        // Copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy-button bg-gray-700 hover:bg-gray-600 text-white text-xs px-2 py-1 rounded flex items-center gap-1 transition-colors';
        copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="m5 15-2-2 2-2"></path>
                <path d="M5 9H3a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-2"></path>
            </svg>
            Copy
        `;
        
        // Edit button
        const editBtn = document.createElement('button');
        editBtn.className = 'code-edit-button bg-gray-700 hover:bg-gray-600 text-white text-xs px-2 py-1 rounded flex items-center gap-1 transition-colors';
        editBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                <path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
            </svg>
            Edit
        `;
        
        buttonsContainer.appendChild(copyBtn);
        buttonsContainer.appendChild(editBtn);
        
        return buttonsContainer;
    }

    // Enhanced copy handler to work with new button structure
    handleCodeCopy(event) {
        if (!event.target.closest('.code-copy-button')) return;

        const codeContainer = event.target.closest('.code-container');
        const codeBlock = codeContainer?.querySelector('pre code');
        
        if (!codeBlock) return;

        const copyButton = event.target.closest('.code-copy-button');
        const originalContent = copyButton.innerHTML;

        navigator.clipboard.writeText(codeBlock.innerText)
            .then(() => {
                copyButton.innerHTML = `
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20,6 9,17 4,12"></polyline>
                    </svg>
                    Copied!
                `;
                copyButton.classList.add('bg-green-600', 'hover:bg-green-500');
                copyButton.classList.remove('bg-gray-700', 'hover:bg-gray-600');
                
                setTimeout(() => {
                    copyButton.innerHTML = originalContent;
                    copyButton.classList.remove('bg-green-600', 'hover:bg-green-500');
                    copyButton.classList.add('bg-gray-700', 'hover:bg-gray-600');
                }, 2000);
            })
            .catch(() => {
                copyButton.innerHTML = `
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                    Error
                `;
            });
    }

    // Add edit handler method
    handleCodeEdit(event) {
        if (!event.target.closest('.code-edit-button')) return;

        const codeContainer = event.target.closest('.code-container');
        const codeBlock = codeContainer?.querySelector('pre code');
        
        if (!codeBlock) return;

        // Create editable textarea
        const originalContent = codeBlock.innerText;
        const textarea = document.createElement('textarea');
        textarea.className = 'w-full h-32 p-3 bg-gray-800 text-gray-100 font-mono text-sm border border-gray-600 rounded resize-none focus:outline-none focus:ring-2 focus:ring-blue-500';
        textarea.value = originalContent;
        
        // Create action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'flex gap-2 mt-2';
        
        const saveBtn = document.createElement('button');
        saveBtn.className = 'px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded transition-colors';
        saveBtn.textContent = 'Save';
        
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white text-xs rounded transition-colors';
        cancelBtn.textContent = 'Cancel';
        
        actionsDiv.appendChild(saveBtn);
        actionsDiv.appendChild(cancelBtn);
        
        // Replace code block with editor
        const pre = codeBlock.parentNode;
        const editContainer = document.createElement('div');
        editContainer.className = 'p-3';
        editContainer.appendChild(textarea);
        editContainer.appendChild(actionsDiv);
        
        pre.style.display = 'none';
        codeContainer.appendChild(editContainer);
        textarea.focus();
        
        // Handle save
        saveBtn.addEventListener('click', () => {
            const newContent = textarea.value;
            // You can add logic here to update the conversation history
            // or send the edited code back to your backend
            
            // For now, just update the display
            const language = codeBlock.className.match(/language-(\w+)/)?.[1] || 'plaintext';
            const highlightedCode = hljs.highlight(newContent, { language }).value;
            codeBlock.innerHTML = highlightedCode;
            
            editContainer.remove();
            pre.style.display = 'block';
        });
        
        // Handle cancel
        cancelBtn.addEventListener('click', () => {
            editContainer.remove();
            pre.style.display = 'block';
        });
    }

    // Fix strong element spacing and highligh code
    fixStyle(root) {
        hljs.highlightAll();
        const paragraphs = root.querySelectorAll('p');

        paragraphs.forEach(p => {
            const firstElement = Array.from(p.childNodes).find(node =>
                node.nodeType === Node.ELEMENT_NODE ||
                (node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '')
            );

            if (firstElement?.nodeType === Node.ELEMENT_NODE && firstElement.tagName === 'STRONG') {
                firstElement.style.display = 'block';
                firstElement.style.marginTop = '1.5rem';
            }
        });
    }

    // Show typing indicator
    showTypingIndicator() {
        this.removeTypingIndicator();

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message flex justify-start';
        typingDiv.id = 'dynamic-typing-indicator';
        typingDiv.innerHTML = `
            <div class="flex-shrink-0 w-10 h-10 rounded-full ai-avatar flex items-center justify-center text-white mr-3">
                <i class="fas fa-robot"></i>
            </div>
            <div class="max-w-[85%] md:max-w-[75%]">
                <div class="assistant-bubble rounded-2xl rounded-tl-none p-4 shadow-sm message-bubble">
                    <div class="typing-indicator flex items-center">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        `;
        
        this.elements.chatContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    // Remove typing indicator
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('dynamic-typing-indicator');
        typingIndicator?.remove();
    }

    // Scroll to bottom
    scrollToBottom() {
        const lastMessage = this.elements.chatContainer.lastElementChild;
        lastMessage?.scrollIntoView({ behavior: 'smooth' });
    }

    // Save chat to localStorage
    saveChatToStorage() {
        const allChats = this.getAllChats();
        const chatId = sessionStorage.getItem(this.config.storageKeys.currentChatId);
        const timestamp = sessionStorage.getItem(this.config.storageKeys.chatStartTimestamp) || new Date().toISOString();

        const updatedChats = allChats.filter(c => c.id !== chatId);
        updatedChats.push({
            id: chatId,
            title: this.generateChatTitle(this.conversationHistory),
            timestamp: timestamp,
            history: this.conversationHistory
        });

        localStorage.setItem(this.config.storageKeys.multiChat, JSON.stringify(updatedChats));
        this.updateSidebarChats();
    }

    // Get all chats from storage
    getAllChats() {
        return JSON.parse(localStorage.getItem(this.config.storageKeys.multiChat) || '[]');
    }

    // Update sidebar with chats
    updateSidebarChats() {
        const allChats = this.getAllChats().sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        this.elements.sidebarContainers.forEach(container => {
            container.innerHTML = '';
            allChats.forEach(chat => {
                container.appendChild(this.createChatHistoryItem(chat));
            });
        });
    }

    // Create chat history item
    createChatHistoryItem(chat) {
        const item = document.createElement('div');
        item.className = 'chat-history-item p-3 rounded-lg cursor-pointer';
        
        const displayDate = new Date(chat.timestamp).toLocaleString([], {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        item.innerHTML = `
            <div class="text-sm font-medium dark:text-gray-200 truncate">${chat.title}</div>
            <div class="text-xs dark:text-gray-400">${displayDate}</div>
        `;

        item.addEventListener('click', () => this.loadChat(chat));
        return item;
    }

    // Load specific chat
    loadChat(chat) {
        sessionStorage.setItem(this.config.storageKeys.currentChatId, chat.id);
        sessionStorage.setItem(this.config.storageKeys.chatStartTimestamp, chat.timestamp);
        
        this.conversationHistory = chat.history;
        this.elements.chatContainer.innerHTML = '';
        
        this.conversationHistory.forEach(msg => {
            this.addMessage(msg.content, msg.role);
        });
    }

    // Load chat from storage
    loadChatFromStorage() {
        this.updateSidebarChats();
        const allChats = this.getAllChats();
        
        if (allChats.length > 0) {
            const latestChat = allChats.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
            this.loadChat(latestChat);
        } else {
            this.displayWelcomeMessage();
        }
        hljs.highlightAll();
    }

    // Generate chat title
    generateChatTitle(history) {
        const userMsg = history.find(msg => msg.role === 'user');
        if (!userMsg) return 'New Chat';
        
        const cleanText = userMsg.content.replace(/\s+/g, ' ').trim();
        return cleanText.length > this.config.maxTitleLength 
            ? cleanText.slice(0, this.config.maxTitleLength) + '...' 
            : cleanText;
    }

    // Display welcome message
    displayWelcomeMessage() {
        const now = new Date();
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message flex justify-start';
        welcomeDiv.innerHTML = `
            <div class="flex-shrink-0 w-10 h-10 rounded-full ai-avatar flex items-center justify-center text-white mr-3">
                <i class="fas fa-robot"></i>
            </div>
            <div class="max-w-[90%] md:max-w-[80%] lg:max-w-[70%] xl:max-w-[60%] 2xl:max-w-[50%]">
                <div class="assistant-bubble rounded-2xl rounded-tl-none p-5 shadow-sm message-bubble">
                    <p class="text-gray-800 dark:text-gray-100">Hello! I'm TensorAlix, your advanced AI assistant. What would you like to discuss in this new conversation?</p>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-2 ml-1 flex items-center">
                    <i class="fas fa-clock mr-1"></i> Today at ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
            </div>
        `;
        this.elements.chatContainer.appendChild(welcomeDiv);
    }

    // Handle page load
    handlePageLoad() {
        const loader = document.getElementById('page-loader');
        const hasVisited = localStorage.getItem(this.config.storageKeys.welcomeShown);

        if (!hasVisited && loader) {
            localStorage.setItem(this.config.storageKeys.welcomeShown, 'true');
            setTimeout(() => {
                loader.style.transition = 'opacity 0.6s ease';
                loader.style.opacity = '0';
                setTimeout(() => loader.remove(), 600);
            }, 2000);
        } else if (loader) {
            loader.remove();
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const chatApp = new ChatApplication();
    chatApp.init();
});