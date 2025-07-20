document.addEventListener('DOMContentLoaded', () => {
    initializeTimeDisplay();
    initializeAttachmentMenu();
    initializeMobileMenu();
    initializeNewChatButtons();
    initializeChatFunctionality();
    loadChatFromLocalStorage();
});

let socket = null;
let chatContainer, messageInput, sendButton;
let conversationHistory = [];
let pendingMessages = [];
let streamingMessageDiv = null;
let streamingMessageContent = '';

const uuid = crypto.randomUUID();

function initializeTimeDisplay() {
    const now = new Date();
    const currentTimeSpan = document.getElementById('current-time');
    if (currentTimeSpan) {
        currentTimeSpan.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
}


function initializeAttachmentMenu() {
    const attachmentButton = document.getElementById('attachment-button');
    const attachmentMenu = document.getElementById('attachment-menu');

    attachmentButton.addEventListener('click', (e) => {
        e.stopPropagation();
        attachmentMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
        attachmentMenu.classList.add('hidden');
    });
}

function initializeMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu');
    const mobileSidebar = document.getElementById('mobile-sidebar');
    const closeMobileMenu = document.getElementById('close-mobile-menu');

    mobileMenuButton.addEventListener('click', () => {
        mobileSidebar.classList.remove('hidden');
        setTimeout(() => {
            mobileSidebar.querySelector('div').classList.remove('-translate-x-full');
        }, 10);
    });

    closeMobileMenu.addEventListener('click', () => {
        mobileSidebar.querySelector('div').classList.add('-translate-x-full');
        setTimeout(() => {
            mobileSidebar.classList.add('hidden');
        }, 300);
    });
}

function initializeNewChatButtons() {
    const newChatButtons = [
        document.getElementById('new-chat'),
        document.getElementById('new-chat-mobile')
    ];

    newChatButtons.forEach(button => {
        button.addEventListener('click', () => {
            const newChatId = crypto.randomUUID();
            const timestamp = new Date().toISOString();
            sessionStorage.setItem('currentChatId', newChatId);
            sessionStorage.setItem('chatStartTimestamp', timestamp);

            conversationHistory = [];
            chatContainer.innerHTML = '';
            disconnectWebSocket();
            displayWelcomeMessage();
        });
    });
}

function initializeChatFunctionality() {
    chatContainer = document.getElementById('chat-container');
    messageInput = document.getElementById('message-input');
    sendButton = document.getElementById('send-button');

    messageInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = `${this.scrollHeight}px`;
    });

    messageInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);
}

function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    conversationHistory.push({ role: 'user', content: message });

    if (!sessionStorage.getItem('currentChatId')) {
        sessionStorage.setItem('currentChatId', crypto.randomUUID());
    }
    if (!sessionStorage.getItem('chatStartTimestamp')) {
        sessionStorage.setItem('chatStartTimestamp', new Date().toISOString());
    }

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        connectWebSocket();
    }

    messageInput.value = '';
    messageInput.style.height = 'auto';

    const payload = JSON.stringify({ conv_id: uuid, history: [...conversationHistory] });

    if (socket.readyState === WebSocket.OPEN) {
        socket.send(payload);
    } else {
        pendingMessages.push(payload);
    }

    showTypingIndicator();
    saveChatToLocalStorage();
}

function connectWebSocket() {
    socket = new WebSocket(`ws://127.0.0.1:8000/ws/${uuid}`);
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
        pendingMessages.forEach(msg => socket.send(msg));
        pendingMessages = [];
    };

    socket.onmessage = (event) => {
        const dynamicTyping = document.getElementById('dynamic-typing-indicator');
        if (dynamicTyping) dynamicTyping.remove();

        try {
            const response = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(event.data)));

            if (response.type === 'stream') {
                if (!streamingMessageDiv) {
                    const uniqueId = `streaming-text-${Date.now()}`;
                    streamingMessageContent = '';
                    streamingMessageDiv = document.createElement('div');
                    streamingMessageDiv.className = 'message flex justify-start';
                    streamingMessageDiv.innerHTML = `
                        <div class="flex-shrink-0 w-10 h-10 rounded-full ai-avatar flex items-center justify-center text-white mr-3">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="max-w-[85%] md:max-w-[75%]">
                            <div class="assistant-bubble rounded-2xl rounded-tl-none p-5 shadow-sm message-bubble">
                                <p class="text-gray-800 dark:text-gray-100" id="${uniqueId}"></p>
                            </div>
                        </div>
                    `;
                    streamingMessageDiv.dataset.streamId = uniqueId;
                    chatContainer.appendChild(streamingMessageDiv);
                    streamingMessageDiv.scrollIntoView({ behavior: 'smooth' });
                }

                streamingMessageContent += response.content;
                const streamId = streamingMessageDiv.dataset.streamId;
                const safeHTML = DOMPurify.sanitize(marked.parse(streamingMessageContent));
                document.getElementById(streamId).innerHTML = safeHTML;
            }
            else if (response.type === 'end_stream') {
                conversationHistory.push({ role: 'assistant', content: streamingMessageContent });
                saveChatToLocalStorage();
                streamingMessageDiv = null;
                streamingMessageContent = '';
            }
            else if (response.type === 'text') {
                conversationHistory.push({ role: 'assistant', content: response.content });
                addMessage(response.content, 'assistant');
            }
            else if (response.type === 'image') {
                const dataUri = `data:image/png;base64,${response.content.trim()}`;
                const imageHtml = `<div class="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 mt-2"><img src="${dataUri}" alt="Generated Image" class="w-full h-auto"/></div>`;
                conversationHistory.push({ role: 'assistant', content: imageHtml });
                addMessage(imageHtml, 'assistant');
            }
        } catch (err) {
            console.error("Failed to parse response:", err);
        }
    };

    socket.onclose = () => { console.warn("WebSocket closed."); };
    socket.onerror = (error) => { console.error("WebSocket error:", error); };
}

function disconnectWebSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close(1000, 'Client disconnecting');
    }
}

function saveChatToLocalStorage() {
    const allChats = JSON.parse(localStorage.getItem('tensoralix-multi-chat') || '[]');
    const chatId = sessionStorage.getItem('currentChatId');
    const timestamp = sessionStorage.getItem('chatStartTimestamp') || new Date().toISOString();

    const updatedChats = allChats.filter(c => c.id !== chatId);

    updatedChats.push({
        id: chatId,
        title: generateChatTitle(conversationHistory),
        timestamp: timestamp,
        history: conversationHistory
    });

    localStorage.setItem('tensoralix-multi-chat', JSON.stringify(updatedChats));
    updateSidebarChats();
}

function updateSidebarChats() {
    const allChats = JSON.parse(localStorage.getItem('tensoralix-multi-chat') || '[]');
    const sidebarContainers = document.querySelectorAll('.sidebar .space-y-1');
    sidebarContainers.forEach(container => container.innerHTML = '');

    allChats.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    allChats.forEach(chat => {
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
        item.addEventListener('click', () => {
            sessionStorage.setItem('currentChatId', chat.id);
            sessionStorage.setItem('chatStartTimestamp', chat.timestamp);
            conversationHistory = chat.history;
            chatContainer.innerHTML = '';
            conversationHistory.forEach(msg => addMessage(msg.content, msg.role));
        });

        sidebarContainers.forEach(container => container.appendChild(item));
    });
}

function loadChatFromLocalStorage() {
    updateSidebarChats();

    const allChats = JSON.parse(localStorage.getItem('tensoralix-multi-chat') || '[]');
    if (allChats.length > 0) {
        const latestChat = allChats.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];

        sessionStorage.setItem('currentChatId', latestChat.id);
        sessionStorage.setItem('chatStartTimestamp', latestChat.timestamp);
        conversationHistory = latestChat.history;
        chatContainer.innerHTML = '';
        conversationHistory.forEach(msg => addMessage(msg.content, msg.role));
    } else {
        // Показываем стандартное приветствие, если нет сохранённых чатов
        displayWelcomeMessage();
    }
}


function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const safeHTML = DOMPurify.sanitize(marked.parse(text));

    messageDiv.innerHTML = sender === 'user'
        ? `<div class="max-w-[85%] md:max-w-[75%]">
            <div class="user-bubble rounded-2xl rounded-tr-none p-5 shadow-sm message-bubble">
                <div class="parsed-markdown">${safeHTML}</div>
            </div>
            <div class="text-xs text-gray-500 dark:text-gray-400 mt-2 mr-1 text-right flex items-center justify-end">
                <i class="fas fa-clock mr-1"></i>${time}
            </div>
        </div>
        <div class="flex-shrink-0 w-10 h-10 rounded-full user-avatar flex items-center justify-center text-white ml-3">
            <i class="fas fa-user"></i>
        </div>`
        : `<div class="flex-shrink-0 w-10 h-10 rounded-full ai-avatar flex items-center justify-center text-white mr-3">
            <i class="fas fa-robot"></i>
        </div>
        <div class="max-w-[85%] md:max-w-[75%]">
            <div class="assistant-bubble rounded-2xl rounded-tl-none p-5 shadow-sm message-bubble">
                <div class="parsed-markdown text-gray-800 dark:text-gray-100">${safeHTML}</div>
            </div>
            <div class="text-xs text-gray-500 dark:text-gray-400 mt-2 ml-1 flex items-center">
                <i class="fas fa-clock mr-1"></i>${time}
            </div>
        </div>`;

    chatContainer.appendChild(messageDiv);
    messageDiv.scrollIntoView({ behavior: 'smooth' });
}

function showTypingIndicator() {
    const oldTyping = document.getElementById('dynamic-typing-indicator');
    if (oldTyping) oldTyping.remove();

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
    chatContainer.appendChild(typingDiv);
    typingDiv.scrollIntoView({ behavior: 'smooth' });
}

function generateChatTitle(history) {
    const userMsg = history.find(msg => msg.role === 'user');
    if (!userMsg) return 'New Chat';
    const cleanText = userMsg.content.replace(/\s+/g, ' ').trim();
    return cleanText.length > 30 ? cleanText.slice(0, 30) + '...' : cleanText;
}

function displayWelcomeMessage() {
    const now = new Date();
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = `message flex justify-start`;
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
    chatContainer.appendChild(welcomeDiv);
}

window.addEventListener('load', () => {
    const loader = document.getElementById('page-loader');

    const hasVisited = localStorage.getItem('tensoralix-welcome-shown');

    if (!hasVisited && loader) {
        localStorage.setItem('tensoralix-welcome-shown', 'true');
        setTimeout(() => {
            loader.style.transition = 'opacity 0.6s ease';
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 600);
        }, 2000);
    } else if (loader) {
        loader.remove();
    }
});
