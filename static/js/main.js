// main.js

import translationService from './translation-service.js';
import EmotionDetector from './emotion-detection.js';

// DOM Elements
const dom = {
    userInput: document.getElementById('user-input'),
    chatMessages: document.getElementById('chat-messages'),
    sendButton: document.getElementById('send-button'),
    themeToggle: document.getElementById('theme-toggle'),
    summaryButton: document.getElementById('summary-btn'),
    exportButton: document.getElementById('export-btn'),
    summaryModal: document.getElementById('summary-modal'),
    closeModalBtn: document.querySelector('.close-button'),
    stressIndicator: document.getElementById('stress-level-indicator'),
    stressText: document.getElementById('stress-level-text'),
    modalTitle: document.getElementById('modal-title'),
    summaryContent: document.getElementById('summary-content'),
    exerciseContent: document.getElementById('exercise-content'),
    exerciseBtn: document.querySelector('[onclick="generateExercise()"]'),
    languageContainer: document.getElementById('language-container'),
    originalMessageToggle: document.getElementById('show-original-toggle'),
    voiceButton: document.getElementById('voice-button'),
    listeningIndicator: document.getElementById('listening-indicator'),
    emotionButton: document.getElementById('emotion-button')
};

// Application State
const state = {
    userId: 'default',
    theme: localStorage.getItem('theme') || 'light',
    isTyping: false,
    useRag: localStorage.getItem('useRag') !== 'false', // Default to true for RAG
    currentLanguage: localStorage.getItem('preferredLanguage') || 'en',
    showOriginalMessages: localStorage.getItem('showOriginalMessages') === 'true',
    lastDetectedLanguage: null,
    isRecording: false,
    mediaRecorder: null, 
    audioChunks: [],
    isSpeaking: false,
    currentAudio: null,
    currentEmotion: null,  
    emotionDetector: null 
};

// Initialize Application
function initializeApp() {
    setupEventListeners();
    loadTheme();
    checkHealth();
    if (dom.languageContainer) {
        translationService.initLanguageSelector('language-container');
    }
    setupTranslationAttributes();
    translationService.setLanguage(state.currentLanguage);
    initializeAudioRecording();

    
    
    // Initialize emotion detector
    import('./emotion-detection.js')
    .then(module => {
        // Get the class constructor
        const EmotionDetectorClass = module.default;
        
        // Create the instance
        state.emotionDetector = new EmotionDetectorClass();
        
        // Initialize it if it loaded properly
        if (state.emotionDetector && typeof state.emotionDetector.init === 'function') {
            state.emotionDetector.init();
        }
    })
    .catch(error => {
        console.error("Error loading emotion detection module:", error);
        
        // Create a simple fallback implementation
        state.emotionDetector = createFallbackEmotionDetector();
    });
}

// Set up translation attributes
function setupTranslationAttributes() {
    const elementsToTranslate = [
        { element: document.querySelector('.profile-section h2'), key: 'title' },
        { element: document.querySelector('.profile-section p'), key: 'subtitle' },
        { element: document.querySelector('.stress-dashboard h3'), key: 'stress_analysis' },
        { element: document.querySelector('.emotion-dashboard h3'), key: 'emotion_analysis' },
        { element: document.querySelector('.emotion-camera-toggle span'), key: 'enable_camera' },
        { element: document.getElementById('emotion-text'), key: 'current_emotion' },
        { element: document.getElementById('theme-toggle'), key: 'toggle_theme' },
        { element: document.querySelector('[onclick="showSummary()"]'), key: 'summary' },
        { element: document.querySelector('[onclick="generateExercise()"]'), key: 'generate_exercise' },
        { element: document.querySelector('#export-pdf-btn'), key: 'export_pdf' },
        { element: document.getElementById('send-button'), key: 'send' }
    ];

    elementsToTranslate.forEach(item => {
        if (item.element) item.element.setAttribute('data-i18n', item.key);
    });

    if (dom.userInput) {
        dom.userInput.setAttribute('data-i18n-placeholder', 'type_message');
    }
}

// Initialize audio recording with MediaRecorder
function initializeAudioRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Audio recording not supported in this browser');
        dom.voiceButton.style.display = 'none';
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            state.mediaRecorder = new MediaRecorder(stream);

            state.mediaRecorder.ondataavailable = (event) => {
                state.audioChunks.push(event.data);
            };

            state.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
                state.audioChunks = [];
                await sendVoiceMessage(audioBlob);
                state.isRecording = false;
                dom.voiceButton.classList.remove('recording');
                dom.listeningIndicator.classList.add('hidden');
                stream.getTracks().forEach(track => track.stop());
            };

            state.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                state.isRecording = false;
                dom.voiceButton.classList.remove('recording');
                dom.listeningIndicator.classList.add('hidden');
                appendMessage('Error recording audio. Please try again.', 'assistant');
            };
        })
        .catch(error => {
            console.error('Error accessing microphone:', error);
            dom.voiceButton.style.display = 'none';
            appendMessage('Microphone access denied or unavailable.', 'assistant');
        });
}

// Event Listeners
function setupEventListeners() {
    dom.userInput.addEventListener('keypress', handleEnterPress);
    dom.sendButton.addEventListener('click', handleSendClick);
    dom.themeToggle.addEventListener('click', toggleTheme);
    dom.summaryButton?.addEventListener('click', showSummary);
    dom.exportButton?.addEventListener('click', exportConversation);
    dom.closeModalBtn?.addEventListener('click', closeModal);
    dom.voiceButton.addEventListener('click', toggleVoiceInput);
    dom.emotionButton?.addEventListener('click', activateEmotionCapture);
    
    // Listen for emotion detection events
    document.addEventListener('emotion-captured', handleEmotionCaptured);
    document.addEventListener('emotion-updated', handleEmotionUpdated);

    window.addEventListener('click', (e) => {
        if (e.target === dom.summaryModal) closeModal();
    });
}

// Emotion detection event handlers
function handleEmotionCaptured(event) {
    if (!event.detail || !event.detail.emotion) return;
    
    state.currentEmotion = event.detail.emotion;
    console.log('Emotion captured:', state.currentEmotion);
}

function handleEmotionUpdated(event) {
    if (!event.detail || !event.detail.emotion) return;
    
    state.currentEmotion = event.detail.emotion;
    console.log('Emotion updated:', state.currentEmotion);
}

// Activate emotion capture
function activateEmotionCapture() {
    if (!state.emotionDetector) {
        appendMessage("Emotion detection is not available. Please try again later.", "assistant");
        return;
    }
    
    try {
        state.emotionDetector.startWebcam();
    } catch (error) {
        console.error('Error activating emotion capture:', error);
        appendMessage('Failed to access camera for emotion detection.', 'assistant');
    }
}



function createFallbackEmotionDetector() {
    return {
        init() { console.log("Using fallback emotion detector"); },
        getEmotionIcon() { return "face-meh"; },
        getEmotionColor() { return "#696969"; },
        startWebcam() { 
            console.warn("Webcam not available in fallback mode");
            appendMessage("Emotion detection is not available. Please check your browser permissions.", "assistant");
            return false;
        },
        clearEmotions() {},
        getEmotionData() { return null; }
    };
}

// Toggle voice input
function toggleVoiceInput() {
    if (!state.mediaRecorder) {
        appendMessage('Voice input is not supported or microphone access is denied.', 'assistant');
        return;
    }

    if (state.isRecording) {
        state.mediaRecorder.stop();
    } else {
        state.audioChunks = [];
        state.mediaRecorder.start();
        state.isRecording = true;
        dom.voiceButton.classList.add('recording');
        dom.listeningIndicator.classList.remove('hidden');
    }
}

// Send voice message to backend with emotion data
async function sendVoiceMessage(audioBlob) {
    showTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'voice_input.webm');
        formData.append('user_id', state.userId);
        formData.append('original_language', state.currentLanguage);
        
        // Add emotion data if available
        if (state.currentEmotion) {
            formData.append('image_data', JSON.stringify(state.currentEmotion));
        }

        const response = await fetch('/voice-chat', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        hideTypingIndicator();

        if (data.error) {
            appendMessage(`Error: ${data.error}`, 'assistant');
            return;
        }

        appendMessage(data.message, 'user', data.original_language, data.emotion_analysis);

        if (data.original_language !== 'en') {
            appendTranslationNote(
                `Your voice message was transcribed and translated from ${getLanguageName(data.original_language)} to English for processing.`
            );
        }

        if (data.original_language !== 'en') {
            const translatedResponse = await translationService.processAssistantResponse(
                data.response,
                data.original_language
            );
            appendMessage(translatedResponse.translatedText, 'assistant', data.original_language);

            const lastMessage = dom.chatMessages.lastElementChild;
            if (lastMessage) {
                lastMessage.dataset.originalText = data.response;
                lastMessage.dataset.translatedText = translatedResponse.translatedText;
            }

            if (translatedResponse.needsTranslation) {
                appendTranslationNote(
                    `Response translated to ${getLanguageName(data.original_language)}. 
                     <button class="toggle-translation-btn" onclick="toggleMessageTranslation(this)">
                       Show original
                     </button>`
                );
            }
        } else {
            appendMessage(data.response, 'assistant', 'en');
        }

        if (data.stress_analysis) {
            updateStressIndicator(data.stress_analysis);
        }
        
        // Reset current emotion after use
        state.currentEmotion = null;
        if (state.emotionDetector) {
            state.emotionDetector.clearEmotions();
        }
    } catch (error) {
        console.error('Error sending voice message:', error);
        hideTypingIndicator();
        appendMessage('Sorry, I encountered an error processing your voice input. Please try again.', 'assistant');
    }
}

// Send text message with emotion data
async function sendMessage() {
    const message = dom.userInput.value.trim();
    if (!message || state.isTyping) return;

    dom.userInput.value = '';
    showTypingIndicator();

    try {
        const processedMessage = await translationService.processUserMessage(message);
        state.lastDetectedLanguage = processedMessage.language;

        // Get emotion data if available
        const emotionData = state.emotionDetector ? state.emotionDetector.getEmotionData() : null;
        
        appendMessage(processedMessage.originalText, 'user', processedMessage.language, emotionData);

        if (processedMessage.needsTranslation) {
            appendTranslationNote(
                `Your message was translated from ${getLanguageName(processedMessage.language)} to English for processing.`
            );
        }

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: processedMessage.translatedText,
                user_id: state.userId,
                original_language: processedMessage.language,
                image_data: emotionData ? JSON.stringify(emotionData) : null
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        hideTypingIndicator();

        if (processedMessage.language !== 'en') {
            const translatedResponse = await translationService.processAssistantResponse(
                data.response,
                processedMessage.language
            );
            appendMessage(translatedResponse.translatedText, 'assistant', processedMessage.language);
            
            const lastMessage = dom.chatMessages.lastElementChild;
            if (lastMessage) {
                lastMessage.dataset.originalText = data.response;
                lastMessage.dataset.translatedText = translatedResponse.translatedText;
            }
            
            if (translatedResponse.needsTranslation) {
                appendTranslationNote(
                    `Response translated to ${getLanguageName(processedMessage.language)}. 
                     <button class="toggle-translation-btn" onclick="toggleMessageTranslation(this)">
                       Show original
                     </button>`
                );
            }
        } else {
            appendMessage(data.response, 'assistant', 'en');
        }

        if (data.stress_analysis) {
            updateStressIndicator(data.stress_analysis);
        }
        
        // Reset current emotion after use
        state.currentEmotion = null;
        if (state.emotionDetector) {
            state.emotionDetector.clearEmotions();
        }

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        appendMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }
}

// Message handling
async function handleEnterPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        await sendMessage();
    }
}

async function handleSendClick() {
    await sendMessage();
}

// Append message with emotion data
function appendMessage(text, sender, language = 'en', emotionData = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.dataset.language = language;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = sender === 'user' ? 'You' : 'Wellness Assistant';

    // Add emotion data if available for user messages
    if (sender === 'user' && emotionData) {
        const emotionWidget = document.createElement('div');
        emotionWidget.className = 'emotion-analysis-widget';
        
        const emotionIcon = document.createElement('i');
        emotionIcon.className = `fas fa-${state.emotionDetector.getEmotionIcon(emotionData.emotion)}`;
        emotionIcon.style.color = state.emotionDetector.getEmotionColor(emotionData.emotion);
        
        const emotionText = document.createElement('span');
        emotionText.textContent = emotionData.emotion;
        
        const confidenceSpan = document.createElement('span');
        confidenceSpan.className = 'emotion-confidence';
        confidenceSpan.textContent = `${Math.round(emotionData.confidence * 100)}%`;
        
        emotionWidget.appendChild(emotionIcon);
        emotionWidget.appendChild(document.createTextNode(' '));
        emotionWidget.appendChild(emotionText);
        emotionWidget.appendChild(document.createTextNode(' '));
        emotionWidget.appendChild(confidenceSpan);
        
        bubble.appendChild(document.createElement('br'));
        bubble.appendChild(emotionWidget);
    }

    if (sender === 'assistant') {
        const controls = document.createElement('div');
        controls.className = 'message-controls';

        if (language !== 'en') {
            const langIndicator = document.createElement('span');
            langIndicator.className = 'language-indicator';
            langIndicator.textContent = `(${getLanguageName(language)})`;
            meta.appendChild(langIndicator);
        }

        const speakButton = document.createElement('button');
        speakButton.className = 'speak-button';
        speakButton.innerHTML = '<i class="fas fa-volume-up"></i>';
        speakButton.title = 'Listen to response';
        speakButton.addEventListener('click', () => speakText(text, language));
        controls.appendChild(speakButton);

        meta.appendChild(controls);
    }

    messageDiv.appendChild(bubble);
    messageDiv.appendChild(meta);
    dom.chatMessages.appendChild(messageDiv);

    scrollToBottom();

    if (sender === 'assistant' && !state.isSpeaking) {
        speakText(text, language);
    }
}

// Append translation note
function appendTranslationNote(noteHTML) {
    const noteDiv = document.createElement('div');
    noteDiv.className = 'translation-note';
    noteDiv.innerHTML = noteHTML;
    dom.chatMessages.appendChild(noteDiv);
    scrollToBottom();
}

// Toggle between original and translated text
function toggleMessageTranslation(button) {
    const messageElement = button.closest('.translation-note').previousElementSibling;
    if (!messageElement) return;

    const bubble = messageElement.querySelector('.message-bubble');
    if (!bubble) return;

    const isShowingTranslation = button.textContent.trim() === 'Show original';

    if (isShowingTranslation) {
        bubble.textContent = messageElement.dataset.originalText;
        button.textContent = 'Show translation';
    } else {
        bubble.textContent = messageElement.dataset.translatedText;
        button.textContent = 'Show original';
    }
}

// Get language name
function getLanguageName(code) {
    const language = translationService.supportedLanguages.find(lang => lang.code === code);
    return language ? language.name : code;
}

// Append exercise to chat
function appendExerciseToChat(exercise, methodNote) {
    const exerciseHtml = `
        <div class="exercise-message-content">
            <h3>${exercise.title}</h3>
            <div class="exercise-meta">
                ${exercise.category} | ${exercise.difficulty} | ${Math.round(exercise.duration / 60)} min
            </div>
            <p>${methodNote}</p>
            <p>Open the exercise details to view the full instructions.</p>
        </div>
        <button class="view-details-btn" onclick="showExerciseDetails()">View Exercise Details</button>
    `;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message exercise-message';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble exercise-bubble';
    bubble.innerHTML = exerciseHtml;

    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = 'Wellness Assistant';

    messageDiv.appendChild(bubble);
    messageDiv.appendChild(meta);
    dom.chatMessages.appendChild(messageDiv);

    scrollToBottom();
}

// Show exercise details
function showExerciseDetails() {
    dom.summaryModal.style.display = 'flex';
    dom.exerciseContent.classList.remove('hidden');
    dom.summaryContent.classList.add('hidden');
    dom.modalTitle.textContent = 'Exercise Details';
}

// Display exercise and summary
function displayExerciseAndSummary(exercise, summary, method) {
    if (!dom.modalTitle || !dom.exerciseContent || !dom.summaryContent || !dom.summaryModal) {
        console.error('Required DOM elements missing');
        return;
    }

    dom.modalTitle.textContent = 'Generated Exercise';

    try {
        let summarySection = summary ? `
            <div class="summary-section">
                <h3>Based on Conversation Summary:</h3>
                <p>${summary}</p>
            </div>
        ` : '';

        const personalizationBadge = method === 'rag' ? '<span class="personalized-badge">Personalized</span>' : '';

        dom.exerciseContent.innerHTML = `
            <div class="exercise-container">
                ${summarySection}
                <div class="exercise-section">
                    <h3>${exercise.title} ${personalizationBadge}</h3>
                    <div class="exercise-metadata">
                        <div class="metadata-item">
                            <strong>Category:</strong> 
                            <span class="category-badge ${exercise.category}">${exercise.category}</span>
                        </div>
                        <div class="metadata-item">
                            <strong>Difficulty:</strong> 
                            <span class="difficulty-badge ${exercise.difficulty}">${exercise.difficulty}</span>
                        </div>
                        <div class="metadata-item">
                            <strong>Duration:</strong> ${Math.round(exercise.duration / 60)} minutes
                        </div>
                    </div>
                    <div class="exercise-instructions">
                        <h4>Instructions:</h4>
                        <ol class="instruction-list">
                            ${Array.isArray(exercise.instructions) ? 
                              exercise.instructions.map(instruction => `<li class="instruction-item">${instruction}</li>`).join('') :
                              '<li>No instructions available</li>'}
                        </ol>
                    </div>
                    <div class="exercise-benefits">
                        <h4>Benefits:</h4>
                        <ul class="benefits-list">
                            ${Array.isArray(exercise.benefits) ? 
                              exercise.benefits.map(benefit => `<li class="benefit-item">${benefit}</li>`).join('') :
                              '<li>No benefits listed</li>'}
                        </ul>
                    </div>
                    <div class="exercise-tags">
                        <h4>Tags:</h4>
                        <div class="tag-container">
                            ${Array.isArray(exercise.tags) ? 
                              exercise.tags.map(tag => `<span class="tag">${tag}</span>`).join('') :
                              '<span class="tag">No tags</span>'}
                        </div>
                    </div>
                </div>
            </div>
        `;

        dom.exerciseContent.classList.remove('hidden');
        dom.summaryContent.classList.add('hidden');
        dom.summaryModal.style.display = 'flex';
    } catch (error) {
        console.error('Error displaying exercise:', error);
        alert('Error displaying exercise content');
    }
}

// Toggle generation method
function toggleGenerationMethod() {
    state.useRag = !state.useRag;
    localStorage.setItem('useRag', state.useRag);
    alert(`Generation method changed to ${state.useRag ? 'Advanced AI (RAG)' : 'Traditional AI'}`);
    closeModal();
}

// Update stress indicator
function updateStressIndicator(analysis) {
    const level = analysis.stress_level;
    const confidence = analysis.confidence * 100;

    dom.stressIndicator.style.width = `${confidence}%`;
    dom.stressIndicator.style.backgroundColor = level === 1 ? 'var(--danger-color)' : 'var(--success-color)';
    dom.stressText.textContent = `Stress Level: ${level === 1 ? 'High' : 'Low'} (${confidence.toFixed(1)}% confidence)`;
}

// Show summary with emotion data
async function showSummary() {
    try {
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.userId })
        });

        if (!response.ok) throw new Error('Failed to get summary');

        const data = await response.json();
        
        // Build emotion distribution if available
        let emotionDistribution = '';
        if (data.stats.emotion_distribution && Object.keys(data.stats.emotion_distribution).length > 0) {
            emotionDistribution = `
                <h3>Emotions Detected</h3>
                <div class="emotion-distribution">
                    ${Object.entries(data.stats.emotion_distribution).map(([emotion, count]) => `
                        <div class="emotion-stat">
                            <span class="emotion-badge emotion-badge-${emotion.toLowerCase()}">
                                <i class="fas fa-${state.emotionDetector ? state.emotionDetector.getEmotionIcon(emotion) : 'face-smile'}"></i> 
                                ${emotion}
                            </span>
                            <span class="emotion-count">${count}x</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
                    
        dom.modalTitle.textContent = 'Chat Summary';
        dom.summaryContent.innerHTML = `
            <div class="summary-section">
                <h3>Conversation Summary</h3>
                <p>${data.summary}</p>
                <h3>Statistics</h3>
                <p>Total Messages: ${data.stats.message_count}</p>
                <p>Your Messages: ${data.stats.user_messages}</p>
                <p>Assistant Responses: ${data.stats.assistant_messages}</p>
                <p>High Stress Responses: ${data.stats.high_stress_count}</p>
                <p>Duration: ${Math.round(data.stats.conversation_duration)} minutes</p>
                ${emotionDistribution}
            </div>
        `;

        dom.summaryContent.classList.remove('hidden');
        dom.exerciseContent.classList.add('hidden');
        dom.summaryModal.style.display = 'flex';
    } catch (error) {
        console.error('Error showing summary:', error);
        alert('Failed to generate conversation summary');
    }
}

// Generate exercise
async function generateExercise() {
    try {
        if (!state.userId) {
            appendMessage('Error: Please log in to generate exercises', 'assistant');
            return;
        }

        appendMessage('Generating a personalized exercise...', 'assistant');

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            appendMessage('Request timed out due to slow server response. Please try again later.', 'assistant');
        }, 30000);

        const response = await fetch('/generate-exercise', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.userId }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate exercise');
        }

        const data = await response.json();
        if (!data.success || !data.exercise) throw new Error('No exercise generated');

        if (state.currentLanguage !== 'en') {
            try {
                data.exercise.title = await translationService.translateText(data.exercise.title, state.currentLanguage, 'en');
                if (Array.isArray(data.exercise.instructions)) {
                    data.exercise.instructions = await Promise.all(
                        data.exercise.instructions.map(instruction =>
                            translationService.translateText(instruction, state.currentLanguage, 'en')
                        )
                    );
                }
                if (Array.isArray(data.exercise.benefits)) {
                    data.exercise.benefits = await Promise.all(
                        data.exercise.benefits.map(benefit =>
                            translationService.translateText(benefit, state.currentLanguage, 'en')
                        )
                    );
                }
            } catch (error) {
                console.warn('Translation of exercise failed:', error);
            }
        }

        const methodNote = 'Personalized using advanced conversation analysis';
        appendExerciseToChat(data.exercise, methodNote);
        displayExerciseAndSummary(data.exercise, '', 'rag');
    } catch (error) {
        console.error('Exercise generation error:', error);
        let errorMessage = error.name === 'AbortError'
            ? 'Request timed out due to slow server response. Please try again later.'
            : `Error: ${error.message || 'An unexpected error occurred'}`;
        appendMessage(errorMessage, 'assistant');

        if (error.name === 'AbortError') {
            setTimeout(() => {
                appendMessage('Attempting to retry the exercise generation...', 'assistant');
                generateExercise();
            }, 5000);
        }
    }
}

// Export PDF with emotion data
async function exportPDF() {
    try {
        appendMessage('Preparing your PDF export...', 'assistant');

        const currentTheme = state.theme || 'light';
        const pdfTheme = currentTheme === 'dark' ? 'dark' : 'default';

        const summaryResponse = await fetch('/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.userId })
        });

        if (!summaryResponse.ok) throw new Error('Failed to get conversation summary');

        const summaryData = await summaryResponse.json();
        const currentDate = new Date().toLocaleString();

        // Create emotion analysis section if available
        let emotionAnalysis = '';
        if (summaryData.stats.emotion_distribution && Object.keys(summaryData.stats.emotion_distribution).length > 0) {
            const emotions = Object.entries(summaryData.stats.emotion_distribution)
                .map(([emotion, count]) => `${emotion}: ${count} instances`)
                .join('\n- ');
                
            emotionAnalysis = `
EMOTION ANALYSIS
---------------
Emotions detected during conversation:
- ${emotions}

`;
        }

        let content = `
MENTAL WELLNESS ASSISTANT - CONVERSATION REPORT
===============================================
Generated on: ${currentDate}

WELLNESS SUMMARY
---------------
${summaryData.summary}

ANALYTICS & INSIGHTS
-------------------
Total Messages: ${summaryData.stats.message_count}
User Messages: ${summaryData.stats.user_messages}
Assistant Responses: ${summaryData.stats.assistant_messages}
High Stress Indicators: ${summaryData.stats.high_stress_count}
Conversation Duration: ${Math.round(summaryData.stats.conversation_duration)} minutes

${emotionAnalysis}
MOOD PATTERNS
------------
${summaryData.stats.high_stress_count > 0 ? 
  `This conversation showed indicators of stress or anxiety (${summaryData.stats.high_stress_count} instances detected).` : 
  'No significant stress indicators were detected in this conversation.'}

RECOMMENDATIONS
--------------
Based on this conversation, consider:
- Regular check-ins with the Wellness Assistant
- Practicing the suggested mindfulness exercises
- Tracking your progress over time
- Sharing insights with a healthcare professional if needed

ABOUT THIS REPORT
----------------
This report was automatically generated by the Mental Wellness Assistant.
It contains a summary of your conversation, analytics, and a full transcript.
This information is confidential and intended for your personal use only.

End of Report - Generated on ${currentDate}
`;

        const response = await fetch('/export-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.userId, content, theme: pdfTheme })
        });

        if (!response.ok) throw new Error('Failed to generate PDF');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `wellness-report-${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        const loadingMsg = document.querySelector('.assistant-message:last-child');
        if (loadingMsg) loadingMsg.remove();

        appendMessage('Your wellness report has been successfully generated and downloaded.', 'assistant');
    } catch (error) {
        console.error('Error exporting PDF:', error);
        appendMessage(`Failed to export PDF: ${error.message}. Please try again.`, 'assistant');
    }
}

// Text-to-speech function
async function speakText(text, language) {
    try {
        if (state.currentAudio) {
            state.currentAudio.pause();
            state.currentAudio = null;
        }

        const response = await fetch('/generate-tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`TTS generation failed: ${errorText}`);
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        state.currentAudio = audio;

        audio.onplay = () => {
            state.isSpeaking = true;
            console.log(`Audio playback started in ${getLanguageName(language)}`);
        };

        audio.onended = () => {
            state.isSpeaking = false;
            URL.revokeObjectURL(audioUrl);
            state.currentAudio = null;
            console.log('Audio playback ended');
        };

        audio.onerror = (event) => {
            state.isSpeaking = false;
            URL.revokeObjectURL(audioUrl);
            state.currentAudio = null;
            console.error('Audio playback error:', event);
            appendMessage('Error playing the audio response. Displaying text instead.', 'assistant');
        };

        audio.play();

    } catch (error) {
        console.error('TTS error:', error);
        appendMessage(`Error generating audio: ${error.message}. Falling back to text.`, 'assistant');
    }
}

// Theme management
function loadTheme() {
    document.body.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', state.theme);
    localStorage.setItem('theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = dom.themeToggle.querySelector('i');
    icon.className = state.theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Utility functions
function showTypingIndicator() {
    if (state.isTyping) return;

    state.isTyping = true;
    const indicator = document.createElement('div');
    indicator.className = 'message assistant-message typing';
    indicator.innerHTML = `
        <div class="message-bubble typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    dom.chatMessages.appendChild(indicator);
    scrollToBottom();
}

function hideTypingIndicator() {
    state.isTyping = false;
    const typingIndicator = document.querySelector('.typing');
    if (typingIndicator) typingIndicator.remove();
}

function scrollToBottom() {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function closeModal() {
    dom.summaryModal.style.display = 'none';
    dom.modalTitle.textContent = 'Chat Summary';
    dom.exerciseContent.classList.add('hidden');
    dom.summaryContent.classList.remove('hidden');
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) console.error('Health check failed');
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initializeApp);

// Global functions
window.toggleMessageTranslation = toggleMessageTranslation;
window.sendMessage = sendMessage;
window.appendMessage = appendMessage;
window.showExerciseDetails = showExerciseDetails;
window.generateExercise = generateExercise;
window.showSummary = showSummary;
window.exportPDF = exportPDF;
window.speakText = speakText;


