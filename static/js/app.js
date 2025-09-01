// StudyForge AI - Web Interface JavaScript
'use strict';

// Utility functions for DOM safety
const $ = (selector) => {
    const element = document.querySelector(selector);
    if (!element) {
        console.warn(`Element not found: ${selector}`);
    }
    return element;
};

const $$ = (selector) => document.querySelectorAll(selector);

// Debounce utility
const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Sanitize HTML to prevent XSS
const sanitizeHTML = (str) => {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
};

// Validate input
const validateMessage = (message) => {
    if (!message || typeof message !== 'string') {
        console.debug('Message validation failed: not a string or empty', { message, type: typeof message });
        return false;
    }
    
    const trimmed = message.trim();
    if (trimmed.length === 0) {
        console.debug('Message validation failed: empty after trim');
        return false;
    }
    
    if (message.length > 4000) {
        console.debug('Message validation failed: too long', { length: message.length });
        return false;
    }
    
    return true;
};

class StudyForgeAI {
    constructor() {
        // Connection management
        this.currentSessionId = null;
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        
        // Configuration
        this.config = {};
        
        // Event controller for cleanup
        this.abortController = new AbortController();
        
        // UI state
        this.isLoading = false;
        this.messageHistory = [];
        
        // Initialize application
        this.init().catch(error => {
            console.error('Failed to initialize StudyForge AI:', error);
            this.showError('Failed to initialize application. Please refresh the page.');
        });
        
        // Start health monitoring
        this.startHealthMonitoring();
    }
    
    async init() {
        try {
            // Show loading state
            this.showLoading(true);
            
            // Load user preferences
            this.loadUserPreferences();
            
            // Load configuration from server
            await this.loadConfig();
            
            // Initialize UI components
            this.initializeUI();
            
            // Load sessions
            await this.loadSessions();
            
            // Apply theme
            const theme = this.getUserPreference('theme') || this.config.theme || 'dark';
            this.applyTheme(theme);
            
            // Setup auto-resize textarea
            this.setupTextareaResize();
            
            // Setup accessibility features
            this.setupAccessibility();
            
            // Setup error handling
            this.setupErrorHandling();
            
            // Setup cleanup on page unload
            this.setupCleanup();
            
            // Hide loading state
            this.showLoading(false);
            
            console.log('StudyForge AI Web Interface initialized successfully');
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize application');
            throw error;
        }
    }
    
    loadUserPreferences() {
        try {
            const preferences = localStorage.getItem('studyforge-preferences');
            this.userPreferences = preferences ? JSON.parse(preferences) : {};
        } catch (error) {
            console.warn('Failed to load user preferences:', error);
            this.userPreferences = {};
        }
    }
    
    saveUserPreferences() {
        try {
            localStorage.setItem('studyforge-preferences', JSON.stringify(this.userPreferences));
        } catch (error) {
            console.warn('Failed to save user preferences:', error);
        }
    }
    
    getUserPreference(key) {
        return this.userPreferences[key];
    }
    
    setUserPreference(key, value) {
        this.userPreferences[key] = value;
        this.saveUserPreferences();
    }
    
    async loadConfig() {
        try {
            const response = await fetch('/api/config', {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                signal: this.createTimeoutSignal(10000) // 10 second timeout
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.config = await response.json();
            
            // Safely update UI with config
            this.updateConfigUI();
            
        } catch (error) {
            console.error('Failed to load config:', error);
            this.config = this.getDefaultConfig();
            this.updateConfigUI();
            this.showError('Failed to load configuration. Using defaults.');
        }
    }
    
    getDefaultConfig() {
        return {
            theme: 'dark',
            auto_search: true,
            search_threshold: 0.7,
            max_search_results: 10,
            timeout_seconds: 300,
            retry_count: 3
        };
    }
    
    updateConfigUI() {
        const updates = [
            { id: 'webSearchToggle', property: 'checked', value: this.config.auto_search },
            { id: 'themeSelect', property: 'value', value: this.config.theme },
            { id: 'autoSearchToggle', property: 'checked', value: this.config.auto_search },
            { id: 'searchThreshold', property: 'value', value: this.config.search_threshold },
            { id: 'thresholdValue', property: 'textContent', value: this.config.search_threshold },
            { id: 'maxResults', property: 'value', value: this.config.max_search_results },
            { id: 'timeoutSeconds', property: 'value', value: this.config.timeout_seconds },
            { id: 'retryCount', property: 'value', value: this.config.retry_count }
        ];
        
        updates.forEach(update => {
            const element = $(update.id);
            if (element) {
                element[update.property] = update.value;
            }
        });
    }
    
    initializeUI() {
        const { signal } = this.abortController;
        
        try {
            // Theme toggle with accessibility
            const themeToggle = $('#themeToggle');
            if (themeToggle) {
                themeToggle.addEventListener('click', () => {
                    this.toggleTheme();
                }, { signal });
            }
            
            // New chat button
            const newChatBtn = $('#newChatBtn');
            if (newChatBtn) {
                newChatBtn.addEventListener('click', () => {
                    this.startNewChat();
                }, { signal });
            }
            
            // Chat form
            const chatForm = $('#chatForm');
            if (chatForm) {
                chatForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.sendMessage();
                }, { signal });
            }
            
            // Message input with enhanced handling
            this.setupMessageInput();
            
            // Document upload functionality
            this.setupDocumentUpload();
            
            // Quick action buttons
            this.setupQuickActions();
            
            // Sidebar toggle (mobile)
            const sidebarToggle = $('#sidebarToggle');
            const sidebar = $('#sidebar');
            if (sidebarToggle && sidebar) {
                sidebarToggle.addEventListener('click', () => {
                    this.toggleSidebar();
                }, { signal });
            }
            
            // Settings modal
            this.setupSettingsModal();
            
            // QR code toggle with accessibility
            this.setupQRCodeToggle();
            
            // Setup keyboard shortcuts
            this.setupKeyboardShortcuts();
            
        } catch (error) {
            console.error('Failed to initialize UI:', error);
            this.showError('Failed to initialize user interface');
        }
    }
    
    setupMessageInput() {
        const messageInput = $('#messageInput');
        const sendButton = $('#sendButton');
        const charCounter = $('#charCounter');
        
        if (!messageInput) return;
        
        const { signal } = this.abortController;
        
        // Enhanced keydown handling
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        }, { signal });
        
        // Debounced input handling with validation
        const handleInput = debounce((e) => {
            const value = e.target.value;
            const charCount = value.length;
            
            // Update character counter
            if (charCounter) {
                charCounter.textContent = `${charCount} / 4000`;
                charCounter.setAttribute('aria-live', 'polite');
            }
            
            // Update send button state
            if (sendButton) {
                sendButton.disabled = !validateMessage(value);
                sendButton.setAttribute('aria-disabled', sendButton.disabled);
            }
            
            // Visual feedback for limit
            const isNearLimit = charCount > 3500;
            const isOverLimit = charCount > 4000;
            
            messageInput.classList.toggle('near-limit', isNearLimit);
            messageInput.classList.toggle('over-limit', isOverLimit);
            
            if (charCounter) {
                charCounter.classList.toggle('near-limit', isNearLimit);
                charCounter.classList.toggle('over-limit', isOverLimit);
            }
            
        }, 150);
        
        messageInput.addEventListener('input', handleInput, { signal });
        
        // Initial state
        if (sendButton) {
            sendButton.disabled = true;
        }
    }
    
    toggleTheme() {
        try {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            this.applyTheme(newTheme);
            this.updateConfig({ theme: newTheme });
            
            // Update button ARIA label
            const themeToggle = $('#themeToggle');
            if (themeToggle) {
                themeToggle.setAttribute('aria-label', `Switch to ${currentTheme} theme`);
            }
            
            // Announce theme change to screen readers
            this.announceToScreenReader(`Switched to ${newTheme} theme`);
            
        } catch (error) {
            console.error('Failed to toggle theme:', error);
        }
    }
    
    toggleSidebar() {
        const sidebar = $('#sidebar');
        const sidebarToggle = $('#sidebarToggle');
        
        if (!sidebar || !sidebarToggle) return;
        
        const isOpen = sidebar.classList.contains('open');
        sidebar.classList.toggle('open');
        
        // Update ARIA attributes
        sidebarToggle.setAttribute('aria-expanded', !isOpen);
        sidebar.setAttribute('aria-hidden', isOpen);
        
        // Manage focus
        if (!isOpen) {
            // Opening sidebar - focus first focusable element
            const firstFocusable = sidebar.querySelector('button, a, input, [tabindex]:not([tabindex="-1"])');
            if (firstFocusable) {
                setTimeout(() => firstFocusable.focus(), 100);
            }
        }
    }
    
    setupSettingsModal() {
        const settingsBtn = $('#settingsBtn');
        const settingsModal = $('#settingsModal');
        const closeSettings = $('#closeSettings');
        const saveSettings = $('#saveSettings');
        const searchThreshold = $('#searchThreshold');
        const thresholdValue = $('#thresholdValue');
        
        const { signal } = this.abortController;
        
        if (settingsBtn && settingsModal) {
            settingsBtn.addEventListener('click', () => {
                this.openModal('settingsModal');
            }, { signal });
        }
        
        if (closeSettings) {
            closeSettings.addEventListener('click', () => {
                this.closeModal('settingsModal');
            }, { signal });
        }
        
        if (saveSettings) {
            saveSettings.addEventListener('click', () => {
                this.saveSettings();
            }, { signal });
        }
        
        if (searchThreshold && thresholdValue) {
            searchThreshold.addEventListener('input', (e) => {
                thresholdValue.textContent = e.target.value;
            }, { signal });
        }
        
        // Close modal on background click
        if (settingsModal) {
            settingsModal.addEventListener('click', (e) => {
                if (e.target === settingsModal) {
                    this.closeModal('settingsModal');
                }
            }, { signal });
        }
    }
    
    setupQRCodeToggle() {
        const qrToggle = $('#qrToggle');
        const qrCode = $('#qrCode');
        
        if (!qrToggle || !qrCode) return;
        
        qrToggle.addEventListener('click', () => {
            const isVisible = qrCode.style.display !== 'none';
            
            qrCode.style.display = isVisible ? 'none' : 'block';
            qrToggle.textContent = isVisible ? 'Show QR Code' : 'Hide QR Code';
            qrToggle.setAttribute('aria-expanded', !isVisible);
            qrCode.setAttribute('aria-hidden', isVisible);
            
        }, { signal: this.abortController.signal });
    }
    
    setupDocumentUpload() {
        const uploadButton = $('#uploadButton');
        const uploadInput = $('#documentUpload');
        
        if (!uploadButton || !uploadInput) return;
        
        const { signal } = this.abortController;
        
        // Upload button click triggers file input
        uploadButton.addEventListener('click', () => {
            uploadInput.click();
        }, { signal });
        
        // Handle file selection
        uploadInput.addEventListener('change', async (e) => {
            const files = e.target.files;
            if (files.length === 0) return;
            
            // Show upload progress
            uploadButton.disabled = true;
            uploadButton.innerHTML = `
                <svg class="upload-spinner" width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" stroke-opacity="0.3"/>
                    <path d="m12 2 10 10-10 10" stroke="currentColor" stroke-width="2" fill="none"/>
                </svg>
                Uploading...
            `;
            
            try {
                for (let i = 0; i < files.length; i++) {
                    await this.uploadDocument(files[i]);
                }
                
                this.showNotification(`Successfully uploaded ${files.length} document(s)`, 'success');
                
            } catch (error) {
                console.error('Upload failed:', error);
                this.showError('Failed to upload document(s). Please try again.');
            } finally {
                // Reset upload button
                uploadButton.disabled = false;
                uploadButton.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" stroke="currentColor" stroke-width="2"/>
                        <polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2"/>
                        <line x1="12" y1="18" x2="12" y2="12" stroke="currentColor" stroke-width="2"/>
                        <polyline points="9,15 12,12 15,15" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    Upload Document
                `;
                
                // Clear input for re-upload of same files
                uploadInput.value = '';
            }
        }, { signal });
        
        // Drag and drop support
        const chatContainer = $('#chatContainer');
        if (chatContainer) {
            chatContainer.addEventListener('dragover', (e) => {
                e.preventDefault();
                chatContainer.classList.add('drag-over');
            }, { signal });
            
            chatContainer.addEventListener('dragleave', (e) => {
                e.preventDefault();
                if (!chatContainer.contains(e.relatedTarget)) {
                    chatContainer.classList.remove('drag-over');
                }
            }, { signal });
            
            chatContainer.addEventListener('drop', async (e) => {
                e.preventDefault();
                chatContainer.classList.remove('drag-over');
                
                const files = Array.from(e.dataTransfer.files);
                if (files.length === 0) return;
                
                // Filter for supported file types
                const supportedFiles = files.filter(file => {
                    const supportedTypes = [
                        'text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/markdown', 'text/html', 'application/json', 'text/csv', 'application/xml', 'text/xml'
                    ];
                    return supportedTypes.includes(file.type) || 
                           file.name.toLowerCase().match(/\.(txt|md|html|json|csv|xml|pdf|docx)$/);
                });
                
                if (supportedFiles.length !== files.length) {
                    this.showError('Some files are not supported. Only PDF, DOCX, TXT, MD, HTML, JSON, CSV, and XML files are allowed.');
                }
                
                if (supportedFiles.length > 0) {
                    try {
                        for (const file of supportedFiles) {
                            await this.uploadDocument(file);
                        }
                        this.showNotification(`Successfully uploaded ${supportedFiles.length} document(s)`, 'success');
                    } catch (error) {
                        console.error('Drag and drop upload failed:', error);
                        this.showError('Failed to upload documents. Please try again.');
                    }
                }
            }, { signal });
        }
    }
    
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);
        if (this.currentSessionId) {
            formData.append('session_id', this.currentSessionId);
        }
        
        const response = await fetch('/api/upload-document', {
            method: 'POST',
            body: formData,
            signal: this.createTimeoutSignal(120000) // 2 minute timeout for uploads
        });
        
        if (!response.ok) {
            const errorText = await response.text().catch(() => 'Unknown error');
            throw new Error(`Upload failed: ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Document uploaded successfully:', result);
        
        return result;
    }
    
    setupQuickActions() {
        const quickActionButtons = $$('.quick-action-btn');
        
        if (quickActionButtons.length === 0) return;
        
        const { signal } = this.abortController;
        
        quickActionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                if (action) {
                    this.insertQuickAction(action);
                }
            }, { signal });
        });
        
        // Add hover effects and tooltips
        quickActionButtons.forEach(button => {
            const action = button.dataset.action;
            const tooltips = {
                '/search': 'Search the web for information',
                '/analyze': 'Analyze uploaded documents',
                '/summarize': 'Summarize content or documents',
                '/explain': 'Get detailed explanations',
                '/compare': 'Compare different concepts or documents',
                '/research': 'Deep research on a topic'
            };
            
            if (tooltips[action]) {
                button.setAttribute('title', tooltips[action]);
                button.setAttribute('aria-label', tooltips[action]);
            }
        });
    }
    
    insertQuickAction(action) {
        const messageInput = $('#messageInput');
        if (!messageInput) return;
        
        // Focus the input first
        messageInput.focus();
        
        // Insert the action command
        const currentValue = messageInput.value.trim();
        const newValue = currentValue ? `${currentValue} ${action} ` : `${action} `;
        
        messageInput.value = newValue;
        
        // Trigger input event to update character count and send button state
        messageInput.dispatchEvent(new Event('input', { bubbles: true }));
        
        // Move cursor to end
        messageInput.setSelectionRange(newValue.length, newValue.length);
        
        // Auto-resize textarea if needed
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        
        // Visual feedback
        this.announceToScreenReader(`Added ${action} command to message`);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter = Send message
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
            
            // Ctrl/Cmd + K = New chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.startNewChat();
            }
            
            // Ctrl/Cmd + U = Upload document
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                const uploadInput = $('#documentUpload');
                if (uploadInput) {
                    uploadInput.click();
                }
            }
            
            // Escape = Close modals
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
            
        }, { signal: this.abortController.signal });
    }
    
    setupTextareaResize() {
        const textarea = $('#messageInput');
        if (!textarea) return;
        
        const resizeTextarea = debounce(() => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }, 50);
        
        textarea.addEventListener('input', resizeTextarea, { 
            signal: this.abortController.signal 
        });
        
        // Handle paste events that might change height
        textarea.addEventListener('paste', () => {
            setTimeout(resizeTextarea, 10);
        }, { signal: this.abortController.signal });
    }
    
    // Utility methods for better UX
    showLoading(show = true) {
        const spinner = $('#loadingSpinner');
        if (spinner) {
            spinner.style.display = show ? 'flex' : 'none';
            spinner.setAttribute('aria-hidden', !show);
        }
    }
    
    showError(message, duration = 5000) {
        console.error('Application error:', message);
        
        const errorContainer = $('#errorNotifications');
        if (!errorContainer) return;
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;
        errorDiv.setAttribute('role', 'alert');
        
        errorContainer.appendChild(errorDiv);
        errorContainer.style.display = 'block';
        errorContainer.setAttribute('aria-hidden', false);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
                if (errorContainer.children.length === 0) {
                    errorContainer.style.display = 'none';
                    errorContainer.setAttribute('aria-hidden', true);
                }
            }
        }, duration);
    }
    
    announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
    
    openModal(modalId) {
        const modal = $(`#${modalId}`);
        if (!modal) return;
        
        modal.style.display = 'block';
        modal.setAttribute('aria-hidden', 'false');
        
        // Focus management
        const firstFocusable = modal.querySelector('button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) {
            setTimeout(() => firstFocusable.focus(), 100);
        }
        
        // Store previously focused element
        this.previousFocus = document.activeElement;
    }
    
    closeModal(modalId) {
        const modal = $(`#${modalId}`);
        if (!modal) return;
        
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
        
        // Restore focus
        if (this.previousFocus) {
            this.previousFocus.focus();
            this.previousFocus = null;
        }
    }
    
    closeAllModals() {
        $$('.modal').forEach(modal => {
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
                modal.setAttribute('aria-hidden', 'true');
            }
        });
        
        if (this.previousFocus) {
            this.previousFocus.focus();
            this.previousFocus = null;
        }
    }
    
    setupAccessibility() {
        // Add skip navigation link functionality
        const skipLink = $('.skip-link');
        const mainContent = $('#main-content');
        
        if (skipLink && mainContent) {
            skipLink.addEventListener('click', (e) => {
                e.preventDefault();
                mainContent.focus();
                mainContent.scrollIntoView();
            }, { signal: this.abortController.signal });
        }
        
        // Update connection status for screen readers
        this.updateConnectionStatus('connecting');
    }
    
    setupErrorHandling() {
        // Global error handler
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.showError('An unexpected error occurred');
        }, { signal: this.abortController.signal });
        
        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.showError('An unexpected error occurred');
            event.preventDefault();
        }, { signal: this.abortController.signal });
    }
    
    setupCleanup() {
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
        
        // Cleanup on page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.handlePageHidden();
            } else {
                this.handlePageVisible();
            }
        }, { signal: this.abortController.signal });
    }
    
    cleanup() {
        try {
            // Disconnect WebSocket
            this.disconnectWebSocket();
            
            // Cancel all ongoing requests
            this.abortController.abort();
            
            // Clear intervals
            if (this.heartbeatInterval) {
                clearInterval(this.heartbeatInterval);
            }
            
            console.log('Cleanup completed');
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
    
    handlePageHidden() {
        // Reduce activity when page is hidden
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
    }
    
    handlePageVisible() {
        // Resume activity when page becomes visible
        if (this.websocket && this.isConnected) {
            this.startHeartbeat();
        }
    }
    
    updateConnectionStatus(status) {
        const statusElement = $('#connectionStatus');
        const statusText = $('#connectionText');
        
        if (!statusElement || !statusText) return;
        
        const statusMap = {
            connected: { text: 'Connected', class: 'connected' },
            connecting: { text: 'Connecting...', class: 'connecting' },
            disconnected: { text: 'Disconnected', class: 'disconnected' },
            reconnecting: { text: 'Reconnecting...', class: 'reconnecting' }
        };
        
        const config = statusMap[status] || statusMap.disconnected;
        
        statusText.textContent = config.text;
        statusElement.className = `connection-status ${config.class}`;
        statusElement.setAttribute('aria-hidden', 'false');
        
        // Auto-hide after 3 seconds if connected
        if (status === 'connected') {
            setTimeout(() => {
                statusElement.setAttribute('aria-hidden', 'true');
            }, 3000);
        }
    }
    
    applyTheme(theme) {
        try {
            document.documentElement.setAttribute('data-theme', theme);
            this.setUserPreference('theme', theme);
            
            // Update config if different
            if (this.config.theme !== theme) {
                this.updateConfig({ theme });
            }
        } catch (error) {
            console.error('Failed to apply theme:', error);
        }
    }
    
    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            const sessionList = document.getElementById('sessionList');
            sessionList.innerHTML = '';
            
            data.sessions.forEach(session => {
                const sessionElement = this.createSessionElement(session);
                sessionList.appendChild(sessionElement);
            });
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }
    
    createSessionElement(session) {
        const div = document.createElement('div');
        div.className = 'session-item';
        div.dataset.sessionId = session.session_id;
        
        div.innerHTML = `
            <div class="session-title">${session.title}</div>
            <div class="session-preview">${session.preview}</div>
        `;
        
        div.addEventListener('click', () => {
            this.loadSession(session.session_id);
        });
        
        return div;
    }
    
    async loadSession(sessionId) {
        try {
            // Update UI
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('active');
            });
            document.querySelector(`[data-session-id="${sessionId}"]`).classList.add('active');
            
            // Load session history
            const response = await fetch(`/api/sessions/${sessionId}/history`);
            const data = await response.json();
            
            this.currentSessionId = sessionId;
            this.displayHistory(data.history);
            
            // Connect websocket for this session
            this.connectWebSocket(sessionId);
            
            // Update chat title
            const firstMessage = data.history.find(msg => msg.role === 'user');
            const title = firstMessage ? 
                (firstMessage.content.length > 30 ? firstMessage.content.substring(0, 30) + '...' : firstMessage.content) :
                'New Chat';
            document.getElementById('chatTitle').textContent = title;
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }
    
    displayHistory(history) {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.innerHTML = '';
        
        // Add welcome message if no history
        if (history.length === 0) {
            messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <div class="ai-avatar">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="message-content">
                        <h3>Welcome to StudyForge AI!</h3>
                        <p>Your intelligent study companion with web search capabilities. I can help you with:</p>
                        <ul>
                            <li>🔍 Research and fact-checking with real-time web search</li>
                            <li>📚 Study assistance and explanations</li>
                            <li>💡 Problem solving and analysis</li>
                            <li>🧠 Memory-enhanced conversations</li>
                        </ul>
                        <p>Ask me anything to get started!</p>
                    </div>
                </div>
            `;
            return;
        }
        
        // Display messages
        history.forEach(message => {
            this.displayMessage(message);
        });
        
        this.scrollToBottom();
    }
    
    displayMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.role}`;
        
        const avatar = message.role === 'user' ? 
            '<div class="message-avatar">U</div>' :
            `<div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>`;
        
        const webSearchIndicator = message.used_web_search ? 
            `<span class="web-search-indicator">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
                    <path d="21 21l-4.35-4.35" stroke="currentColor" stroke-width="2"/>
                </svg>
                Web Search
            </span>` : '';
        
        const responseTime = message.response_time ? 
            `<span>${(message.response_time).toFixed(2)}s</span>` : '';
        
        messageElement.innerHTML = `
            ${avatar}
            <div class="message-content">
                <div class="message-bubble">${this.formatMessage(message.content)}</div>
                <div class="message-meta">
                    ${webSearchIndicator}
                    ${responseTime}
                    <span>${this.formatTimestamp(message.timestamp)}</span>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
    }
    
    formatMessage(content) {
        // Basic markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString();
    }
    
    startNewChat() {
        this.currentSessionId = null;
        this.disconnectWebSocket();
        
        // Clear active session
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Clear messages
        document.getElementById('chatMessages').innerHTML = `
            <div class="welcome-message">
                <div class="ai-avatar">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="message-content">
                    <h3>Welcome to StudyForge AI!</h3>
                    <p>Your intelligent study companion with web search capabilities. I can help you with:</p>
                    <ul>
                        <li>🔍 Research and fact-checking with real-time web search</li>
                        <li>📚 Study assistance and explanations</li>
                        <li>💡 Problem solving and analysis</li>
                        <li>🧠 Memory-enhanced conversations</li>
                    </ul>
                    <p>Ask me anything to get started!</p>
                </div>
            </div>
        `;
        
        // Reset title
        document.getElementById('chatTitle').textContent = 'New Chat';
        
        // Focus input
        document.getElementById('messageInput').focus();
    }
    
    async sendMessage() {
        if (this.isLoading) return; // Prevent double sending
        
        const messageInput = $('#messageInput');
        if (!messageInput) return;
        
        const message = messageInput.value.trim();
        
        // Validate message
        if (!validateMessage(message)) {
            console.error('Message validation failed:', { 
                originalValue: messageInput.value, 
                trimmedMessage: message, 
                length: message.length 
            });
            this.showError('Please enter a valid message (1-4000 characters)');
            messageInput.focus();
            return;
        }
        
        console.log('Sending message:', { message, length: message.length, sessionId: this.currentSessionId });
        
        this.isLoading = true;
        
        try {
            // Clear input and update UI
            this.clearMessageInput();
            
            // Display user message
            const userMessage = {
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            };
            
            this.displayMessage(userMessage);
            this.messageHistory.push(userMessage);
            
            // Show loading state
            this.showTypingIndicator(true);
            this.scrollToBottom();
            
            // Send message
            await this.sendMessageWithFallback(message);
            
        } catch (error) {
            console.error('Failed to send message:', error);
            this.displayErrorMessage('Failed to send message. Please try again.');
            
            // Restore message in input for retry
            if (messageInput) {
                messageInput.value = message;
                this.updateCharacterCount(message.length);
            }
            
        } finally {
            this.isLoading = false;
            this.showTypingIndicator(false);
            this.updateSendButtonState();
            
            // Restore focus
            if (messageInput) {
                messageInput.focus();
            }
        }
    }
    
    async sendMessageWithFallback(message) {
        // Validate message first
        const trimmedMessage = message ? message.trim() : '';
        if (!trimmedMessage) {
            throw new Error('Cannot send empty message');
        }
        
        let websocketAttempted = false;
        let apiAttempted = false;
        let lastError = null;
        
        try {
            // Try WebSocket first if available and connected
            if (this.websocket && this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
                websocketAttempted = true;
                console.log('Attempting WebSocket send...');
                await this.sendMessageViaWebSocket(trimmedMessage);
                console.log('WebSocket send successful');
                return; // Success, no need for fallback
            }
        } catch (error) {
            console.warn('WebSocket failed:', error);
            lastError = error;
        }
        
        try {
            // Try API fallback (either WebSocket failed or wasn't available)
            apiAttempted = true;
            console.log('Attempting API send...');
            await this.sendMessageViaAPI(trimmedMessage);
            console.log('API send successful');
            return; // Success
        } catch (error) {
            console.error('API send failed:', error);
            lastError = error;
        }
        
        // Both methods failed, throw the most recent error
        const methodsAttempted = [];
        if (websocketAttempted) methodsAttempted.push('WebSocket');
        if (apiAttempted) methodsAttempted.push('API');
        
        throw new Error(`Failed to send message via ${methodsAttempted.join(' and ')}: ${lastError?.message || 'Unknown error'}`);
    }
    
    startHealthMonitoring() {
        // Check health every 2 minutes
        this.healthCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/health', {
                    method: 'GET',
                    signal: this.createTimeoutSignal(10000) // 10 second timeout for health check
                });
                
                if (response.ok) {
                    const health = await response.json();
                    
                    // Check if AI model is healthy
                    const aiModelHealth = health.components?.find(c => c.name === 'ai_model');
                    
                    if (aiModelHealth && aiModelHealth.status !== 'healthy') {
                        console.warn('AI model health degraded:', aiModelHealth.message);
                        this.showWarning(`AI model status: ${aiModelHealth.message}`);
                    }
                } else {
                    console.warn('Health check failed:', response.status);
                }
                
            } catch (error) {
                // Don't spam console with health check errors, but log them
                if (error.name !== 'TimeoutError' && error.name !== 'AbortError') {
                    console.debug('Health check error (will retry):', error.message);
                }
            }
        }, 120000); // Every 2 minutes
    }
    
    showWarning(message) {
        // Create a subtle warning notification
        const warningElement = document.createElement('div');
        warningElement.className = 'health-warning';
        warningElement.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: var(--warning-color, #f39c12);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 10000;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        warningElement.textContent = `⚠️ ${message}`;
        
        document.body.appendChild(warningElement);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (warningElement.parentNode) {
                warningElement.parentNode.removeChild(warningElement);
            }
        }, 10000);
    }
    
    createTimeoutSignal(timeoutMs) {
        // Create a timeout signal that's compatible across browsers
        if (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) {
            // Modern browsers support AbortSignal.timeout
            return AbortSignal.timeout(timeoutMs);
        } else {
            // Fallback for older browsers
            const controller = new AbortController();
            setTimeout(() => controller.abort(), timeoutMs);
            return controller.signal;
        }
    }
    
    handleStatusMessage(message) {
        // Show status messages (like "🔍 Searching the web...")
        this.showStatusMessage(message);
    }
    
    handleStreamingChunk(data) {
        // Handle streaming text chunks
        if (data.chunk) {
            this.appendToCurrentResponse(data.chunk);
        }
    }
    
    handleStreamingComplete(data) {
        // Handle completion of streaming
        this.showTypingIndicator(false);
        this.finalizeStreamingResponse(data);
    }
    
    showStatusMessage(message) {
        const chatMessages = $('#chatMessages');
        if (!chatMessages) return;
        
        // Remove existing status message
        const existingStatus = chatMessages.querySelector('.status-message');
        if (existingStatus) {
            existingStatus.remove();
        }
        
        // Add new status message
        const statusElement = document.createElement('div');
        statusElement.className = 'status-message';
        statusElement.style.cssText = `
            padding: 8px 12px;
            margin: 8px 0;
            background: var(--accent-color, #007bff);
            color: white;
            border-radius: 12px;
            font-size: 14px;
            text-align: center;
            opacity: 0.8;
        `;
        statusElement.textContent = message;
        
        chatMessages.appendChild(statusElement);
        this.scrollToBottom();
    }
    
    appendToCurrentResponse(chunk) {
        const chatMessages = $('#chatMessages');
        if (!chatMessages) return;
        
        // Find or create streaming response container
        let streamingResponse = chatMessages.querySelector('.streaming-response');
        if (!streamingResponse) {
            streamingResponse = document.createElement('div');
            streamingResponse.className = 'message assistant streaming-response';
            streamingResponse.innerHTML = `
                <div class="message-avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                        <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="currentColor" stroke-width="2"/>
                        <line x1="9" y1="9" x2="9.01" y2="9" stroke="currentColor" stroke-width="2"/>
                        <line x1="15" y1="9" x2="15.01" y2="9" stroke="currentColor" stroke-width="2"/>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="response-text"></div>
                        <div class="typing-cursor">|</div>
                    </div>
                </div>
            `;
            
            // Remove status message when starting response
            const statusMessage = chatMessages.querySelector('.status-message');
            if (statusMessage) {
                statusMessage.remove();
            }
            
            chatMessages.appendChild(streamingResponse);
        }
        
        // Append chunk to response
        const responseText = streamingResponse.querySelector('.response-text');
        if (responseText) {
            responseText.textContent += chunk;
            this.scrollToBottom();
        }
    }
    
    finalizeStreamingResponse(data) {
        const chatMessages = $('#chatMessages');
        const streamingResponse = chatMessages?.querySelector('.streaming-response');
        
        if (streamingResponse) {
            // Remove typing cursor
            const cursor = streamingResponse.querySelector('.typing-cursor');
            if (cursor) {
                cursor.remove();
            }
            
            // Remove streaming class
            streamingResponse.classList.remove('streaming-response');
            
            // Add response metadata
            const responseText = streamingResponse.querySelector('.response-text');
            if (responseText && data.response_time) {
                const metadata = document.createElement('div');
                metadata.className = 'response-metadata';
                metadata.style.cssText = 'font-size: 12px; opacity: 0.7; margin-top: 8px;';
                metadata.textContent = `Response time: ${data.response_time?.toFixed(1)}s`;
                streamingResponse.querySelector('.message-bubble').appendChild(metadata);
            }
        }
        
        // Update message history
        if (data.response) {
            this.messageHistory.push({
                role: 'assistant',
                content: data.response,
                timestamp: new Date().toISOString()
            });
        }
        
        this.scrollToBottom();
    }
    
    clearMessageInput() {
        const messageInput = $('#messageInput');
        const charCounter = $('#charCounter');
        const sendButton = $('#sendButton');
        
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }
        
        if (charCounter) {
            charCounter.textContent = '0 / 4000';
        }
        
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.classList.add('loading');
        }
    }
    
    updateCharacterCount(count) {
        const charCounter = $('#charCounter');
        if (charCounter) {
            charCounter.textContent = `${count} / 4000`;
        }
    }
    
    updateSendButtonState() {
        const sendButton = $('#sendButton');
        const messageInput = $('#messageInput');
        
        if (sendButton && messageInput) {
            const hasValidMessage = validateMessage(messageInput.value);
            sendButton.disabled = !hasValidMessage || this.isLoading;
            sendButton.classList.toggle('loading', this.isLoading);
        }
    }
    
    showTypingIndicator(show = true) {
        const typingIndicator = $('#typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = show ? 'flex' : 'none';
            typingIndicator.setAttribute('aria-hidden', !show);
        }
    }
    
    async sendMessageViaAPI(message) {
        // Validate message again before API call
        const trimmedMessage = message ? message.trim() : '';
        if (!trimmedMessage) {
            throw new Error('Cannot send empty message to API');
        }
        
        const webSearchToggle = $('#webSearchToggle');
        const forceWebSearch = webSearchToggle ? webSearchToggle.checked : false;
        
        const requestBody = {
            message: sanitizeHTML(trimmedMessage),
            force_web_search: forceWebSearch,
            session_id: this.currentSessionId
        };
        
        console.log('API request body:', requestBody);
        
        const timeoutMs = (this.config.timeout_seconds || 300) * 1000; // 5 minute default timeout
        
        // Create abort controller for timeout (more compatible than AbortSignal.timeout)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                },
                body: JSON.stringify(requestBody),
                signal: controller.signal,
                mode: 'cors',
                credentials: 'omit'
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                const errorText = await response.text().catch(() => 'Unknown error');
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
        
        // Handle new session creation
        if (!this.currentSessionId && data.session_id) {
            this.currentSessionId = data.session_id;
            this.connectWebSocket(data.session_id);
            this.updateChatTitle(message);
            
            // Reload sessions to show new chat
            this.loadSessions().catch(error => {
                console.warn('Failed to reload sessions:', error);
            });
        }
        
        // Display AI response
        if (data.response) {
            const aiMessage = {
                role: 'assistant',
                content: data.response,
                timestamp: data.timestamp || new Date().toISOString(),
                response_time: data.response_time || 0,
                used_web_search: data.used_web_search || false,
                error: data.error
            };
            
            this.displayMessage(aiMessage);
            this.messageHistory.push(aiMessage);
            this.scrollToBottom();
        }
        
            // Show error if present
            if (data.error) {
                this.showError(`AI Error: ${data.error}`);
            }
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error(`Request timed out after ${timeoutMs / 1000} seconds`);
            }
            throw error;
        }
    }
    
    updateChatTitle(message) {
        const chatTitle = $('#chatTitle');
        if (chatTitle) {
            const title = message.length > 30 ? message.substring(0, 30) + '...' : message;
            chatTitle.textContent = title;
        }
    }
    
    async sendMessageViaWebSocket(message) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }
        
        // Validate message before sending
        const trimmedMessage = message ? message.trim() : '';
        if (!trimmedMessage) {
            throw new Error('Cannot send empty message via WebSocket');
        }
        
        const webSearchToggle = $('#webSearchToggle');
        const forceWebSearch = webSearchToggle ? webSearchToggle.checked : false;
        
        const messageData = {
            message: sanitizeHTML(trimmedMessage),
            force_web_search: forceWebSearch,
            timestamp: new Date().toISOString()
        };
        
        // Validate message data before sending
        if (!messageData.message || messageData.message.trim() === '') {
            throw new Error('Message data is empty after sanitization');
        }
        
        try {
            const jsonMessage = JSON.stringify(messageData);
            console.log('Sending WebSocket message:', jsonMessage);
            this.websocket.send(jsonMessage);
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            throw error;
        }
    }
    
    connectWebSocket(sessionId) {
        if (this.websocket) {
            this.disconnectWebSocket();
        }
        
        // Validate session ID
        if (!sessionId || typeof sessionId !== 'string' || sessionId.trim() === '') {
            console.error('Cannot connect WebSocket: invalid session ID:', sessionId);
            this.updateConnectionStatus('disconnected');
            return;
        }
        
        this.updateConnectionStatus('connecting');
        this.currentSessionId = sessionId; // Store for reconnection
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
        
        try {
            console.log('Connecting to WebSocket:', wsUrl);
            this.websocket = new WebSocket(wsUrl);
            
            // Add connection timeout
            const connectionTimeout = setTimeout(() => {
                if (this.websocket && this.websocket.readyState === WebSocket.CONNECTING) {
                    console.warn('WebSocket connection timeout');
                    this.websocket.close();
                    this.handleConnectionFailure(sessionId);
                }
            }, 10000); // 10 second timeout
            
            this.websocket.onopen = () => {
                clearTimeout(connectionTimeout);
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.startHeartbeat();
                console.log('WebSocket connected successfully to:', wsUrl);
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.websocket.onclose = (event) => {
                clearTimeout(connectionTimeout);
                this.handleWebSocketClose(event, sessionId);
            };
            
            this.websocket.onerror = (error) => {
                clearTimeout(connectionTimeout);
                this.handleWebSocketError(error);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            this.handleConnectionFailure(sessionId);
        }
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'typing':
                    this.handleTypingIndicator(data.status);
                    break;
                    
                case 'status':
                    this.handleStatusMessage(data.message);
                    break;
                    
                case 'chunk':
                    this.handleStreamingChunk(data);
                    break;
                    
                case 'complete':
                case 'final':
                    this.handleStreamingComplete(data);
                    break;
                    
                case 'response':
                    this.handleAIResponse(data);
                    break;
                    
                case 'error':
                    console.error('WebSocket error received:', data.error);
                    this.showError(`WebSocket Error: ${data.error}`);
                    this.showTypingIndicator(false);
                    // For critical errors like empty message, also disconnect
                    if (data.error.includes('Empty message')) {
                        console.warn('Empty message error - WebSocket may be sending corrupted data');
                    }
                    break;
                    
                case 'info':
                    console.info('WebSocket info:', data.message);
                    break;
                    
                case 'pong':
                    // Heartbeat response
                    break;
                    
                default:
                    console.warn('Unknown WebSocket message type:', data.type);
            }
            
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }
    
    handleTypingIndicator(status) {
        const typingIndicator = $('#typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = status ? 'flex' : 'none';
            typingIndicator.setAttribute('aria-hidden', !status);
        }
    }
    
    handleAIResponse(data) {
        const aiMessage = {
            role: 'assistant',
            content: data.response,
            timestamp: new Date().toISOString(),
            response_time: data.response_time,
            used_web_search: data.used_web_search
        };
        
        this.displayMessage(aiMessage);
        this.scrollToBottom();
        
        // Update message history
        this.messageHistory.push(aiMessage);
    }
    
    handleWebSocketClose(event, sessionId) {
        this.isConnected = false;
        this.stopHeartbeat();
        
        if (event.wasClean) {
            console.log('WebSocket connection closed cleanly');
            this.updateConnectionStatus('disconnected');
        } else {
            console.warn('WebSocket connection closed unexpectedly:', event.code, event.reason);
            this.attemptReconnection(sessionId);
        }
    }
    
    handleWebSocketError(error) {
        console.error('WebSocket error:', error);
        this.isConnected = false;
        this.updateConnectionStatus('error');
        
        // Show user-friendly error message
        this.showError('Connection error occurred. Messages will use slower API fallback.');
    }
    
    handleConnectionFailure(sessionId) {
        console.warn('WebSocket connection failed for session:', sessionId);
        this.isConnected = false;
        this.updateConnectionStatus('failed');
        
        // Don't show error immediately, let API fallback handle it silently
        console.info('WebSocket unavailable, will use API fallback for messages');
    }
    
    attemptReconnection(sessionId) {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('disconnected');
            this.showError('Connection lost. Please refresh the page.');
            return;
        }
        
        this.reconnectAttempts++;
        this.updateConnectionStatus('reconnecting');
        
        // Exponential backoff
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            if (!this.isConnected) {
                this.connectWebSocket(sessionId);
            }
        }, delay);
    }
    
    startHeartbeat() {
        this.stopHeartbeat();
        
        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    disconnectWebSocket() {
        this.stopHeartbeat();
        
        if (this.websocket) {
            this.websocket.onclose = null; // Prevent reconnection attempts
            this.websocket.close(1000, 'Client disconnecting');
            this.websocket = null;
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        }
    }
    
    displayErrorMessage(error) {
        console.error('Displaying error message:', error);
        
        const messagesContainer = document.getElementById('chatMessages');
        const errorElement = document.createElement('div');
        errorElement.className = 'message assistant error-message';
        errorElement.innerHTML = `
            <div class="message-avatar" style="background: var(--error-color);">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" stroke-width="2"/>
                    <line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" stroke-width="2"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble" style="background: var(--error-color); color: white;">
                    ❌ Error: ${error}
                </div>
                <div class="message-meta">
                    <span>now</span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(errorElement);
        this.scrollToBottom();
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    async saveSettings() {
        const newConfig = {
            theme: document.getElementById('themeSelect').value,
            auto_search: document.getElementById('autoSearchToggle').checked,
            search_threshold: parseFloat(document.getElementById('searchThreshold').value),
            max_search_results: parseInt(document.getElementById('maxResults').value),
            timeout_seconds: parseInt(document.getElementById('timeoutSeconds').value),
            retry_count: parseInt(document.getElementById('retryCount').value)
        };
        
        await this.updateConfig(newConfig);
        
        // Apply theme immediately
        this.applyTheme(newConfig.theme);
        
        // Update web search toggle
        document.getElementById('webSearchToggle').checked = newConfig.auto_search;
        
        // Close modal
        document.getElementById('settingsModal').style.display = 'none';
        
        // Show confirmation
        this.showNotification('Settings saved successfully!');
    }
    
    async updateConfig(updates) {
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(updates)
            });
            
            if (response.ok) {
                this.config = { ...this.config, ...updates };
            }
        } catch (error) {
            console.error('Failed to update config:', error);
        }
    }
    
    showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success-color);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            z-index: 1001;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.studyForgeAI = new StudyForgeAI();
});

// Add some additional CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);