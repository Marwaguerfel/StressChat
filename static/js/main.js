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
    exerciseBtn: document.querySelector('[onclick="generateExercise()"]')
};

// Application State
const state = {
    userId: 'default',
    theme: localStorage.getItem('theme') || 'light',
    isTyping: false,
    useRag: localStorage.getItem('useRag') === 'false' ? false : true // Par défaut, utiliser RAG
};

// Initialize Application
function initializeApp() {
    setupEventListeners();
    loadTheme();
    checkHealth();
}

// Event Listeners
function setupEventListeners() {
    // Message input handling
    dom.userInput.addEventListener('keypress', handleEnterPress);
    dom.sendButton.addEventListener('click', handleSendClick);
    
    // Theme toggle
    dom.themeToggle.addEventListener('click', toggleTheme);
    
    // Modal controls
    dom.summaryButton?.addEventListener('click', showSummary);
    dom.exportButton?.addEventListener('click', exportConversation);
    dom.closeModalBtn?.addEventListener('click', closeModal);
    
    // Close modal on outside click
    window.addEventListener('click', (e) => {
        if (e.target === dom.summaryModal) {
            closeModal();
        }
    });
}

// Message Handling
async function handleEnterPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        await sendMessage();
    }
}

async function handleSendClick() {
    await sendMessage();
}

async function sendMessage() {
    const message = dom.userInput.value.trim();
    if (!message || state.isTyping) return;

    // Clear input and show typing
    dom.userInput.value = '';
    showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                user_id: state.userId
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        
        hideTypingIndicator();
        appendMessage(message, 'user');
        appendMessage(data.response, 'assistant');
        
        if (data.stress_analysis) {
            updateStressIndicator(data.stress_analysis);
        }

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        appendMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }
}

function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = sender === 'user' ? 'You' : 'Wellness Assistant';
    
    messageDiv.appendChild(bubble);
    messageDiv.appendChild(meta);
    dom.chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
}









// Fonction pour afficher l'exercice dans le chat
function appendExerciseToChat(exercise, methodNote) {
    // Créer le HTML pour l'exercice
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
    
    // Créer l'élément du message
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

// Fonction pour afficher les détails de l'exercice dans la modal
function showExerciseDetails() {
    dom.summaryModal.style.display = 'flex';
    dom.exerciseContent.classList.remove('hidden');
    dom.summaryContent.classList.add('hidden');
    dom.modalTitle.textContent = 'Exercise Details';
}


function displayExerciseAndSummary(exercise, summary, method) {
    console.log('Displaying exercise:', exercise);
    console.log('Summary:', summary);
    
    // Check if elements exist
    if (!dom.modalTitle || !dom.exerciseContent || !dom.summaryContent || !dom.summaryModal) {
        console.error('Required DOM elements missing:', {
            modalTitle: !!dom.modalTitle,
            exerciseContent: !!dom.exerciseContent,
            summaryContent: !!dom.summaryContent,
            summaryModal: !!dom.summaryModal
        });
        return;
    }

    dom.modalTitle.textContent = 'Generated Exercise';
    
    try {
        let summarySection = '';
        
        if (summary) {
            summarySection = `
                <div class="summary-section">
                    <h3>Based on Conversation Summary:</h3>
                    <p>${summary}</p>
                </div>
            `;
        }
        
        // Add personalization badges if the exercise is from RAG
        const personalizationBadge = method === 'rag' ? 
            '<span class="personalized-badge">Personalized</span>' : '';
        
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
                              exercise.instructions.map(instruction => 
                                `<li class="instruction-item">${instruction}</li>`).join('') :
                              '<li>No instructions available</li>'}
                        </ol>
                    </div>
                    <div class="exercise-benefits">
                        <h4>Benefits:</h4>
                        <ul class="benefits-list">
                            ${Array.isArray(exercise.benefits) ? 
                              exercise.benefits.map(benefit => 
                                `<li class="benefit-item">${benefit}</li>`).join('') :
                              '<li>No benefits listed</li>'}
                        </ul>
                    </div>
                    <div class="exercise-tags">
                        <h4>Tags:</h4>
                        <div class="tag-container">
                            ${Array.isArray(exercise.tags) ? 
                              exercise.tags.map(tag => 
                                `<span class="tag">${tag}</span>`).join('') :
                              '<span class="tag">No tags</span>'}
                        </div>
                    </div>
                </div>
            </div>
        `;
        console.log('Exercise content set successfully');

        dom.exerciseContent.classList.remove('hidden');
        dom.summaryContent.classList.add('hidden');
        dom.summaryModal.style.display = 'flex';
        console.log('Modal should now be visible');

    } catch (error) {
        console.error('Error displaying exercise:', error);
        alert('Error displaying exercise content');
    }
}

// Toggle between RAG and traditional method
function toggleGenerationMethod() {
    state.useRag = !state.useRag;
    localStorage.setItem('useRag', state.useRag);
    
    // Feedback to user
    alert(`Generation method changed to ${state.useRag ? 'Advanced AI (RAG)' : 'Traditional AI'}`);
    
    // Close modal
    closeModal();
}





// Stress Level Indicator
function updateStressIndicator(analysis) {
    const level = analysis.stress_level;
    const confidence = analysis.confidence * 100;
    
    dom.stressIndicator.style.width = `${confidence}%`;
    dom.stressIndicator.style.backgroundColor = 
        level === 1 ? 'var(--danger-color)' : 'var(--success-color)';
    
    dom.stressText.textContent = 
        `Stress Level: ${level === 1 ? 'High' : 'Low'} (${confidence.toFixed(1)}% confidence)`;
}

// Summary and Export Functions
async function showSummary() {
    try {
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: state.userId })
        });

        if (!response.ok) throw new Error('Failed to get summary');
        
        const data = await response.json();
        
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













async function generateExercise() {
    try {
        if (!state.userId) {
            appendMessage('Error: Please log in to generate exercises', 'assistant');
            return;
        }

        appendMessage('Generating a personalized exercise...', 'assistant');

        const controller = new AbortController();
        // Increase timeout to 30 seconds (30000 ms) to allow more time for server response
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

        clearTimeout(timeoutId); // Clear the timeout if the request succeeds

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate exercise');
        }

        const data = await response.json();
        if (!data.success || !data.exercise) {
            throw new Error('No exercise generated');
        }

        const methodNote = 'Personalized using advanced conversation analysis';
        appendExerciseToChat(data.exercise, methodNote);
        displayExerciseAndSummary(data.exercise, '', 'rag');

    } catch (error) {
        console.error('Exercise generation error:', error);
        let errorMessage = 'An unexpected error occurred. Please try again later.';
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out due to slow server response. Please try again later.';
        } else if (error.message) {
            errorMessage = `Error: ${error.message}`;
        }
        appendMessage(errorMessage, 'assistant');

        // Optional: Add retry logic
        if (error.name === 'AbortError') {
            setTimeout(() => {
                appendMessage('Attempting to retry the exercise generation...', 'assistant');
                generateExercise(); // Retry after a delay (e.g., 5 seconds)
            }, 5000);
        }
    }
};


















async function exportPDF() {
    try {
        // Show loading indicator
        const loadingMessage = 'Preparing your PDF export...';
        appendMessage(loadingMessage, 'assistant');
        
        // Get user's current theme preference
        const currentTheme = state.theme || 'light';
        const pdfTheme = currentTheme === 'dark' ? 'dark' : 'default';
        
        // Fetch conversation summary data
        const summaryResponse = await fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: state.userId })
        });

        if (!summaryResponse.ok) {
            throw new Error('Failed to get conversation summary');
        }

        // Process summary data
        const summaryData = await summaryResponse.json();
        const messages = dom.chatMessages.children;
        const currentDate = new Date().toLocaleString();
        
        // Create formatted PDF content with enhanced sections and formatting
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
        
        // Send request to generate PDF with the formatted content and theme preference
        const response = await fetch('/export-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: state.userId,
                content: content,
                theme: pdfTheme
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('PDF generation failed:', errorText);
            throw new Error('Failed to generate PDF');
        }

        // Download the file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `wellness-report-${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Remove loading message and show confirmation
        const loadingMsg = document.querySelector('.assistant-message:last-child');
        if (loadingMsg) {
            loadingMsg.remove();
        }
        
        // Success message with information about the color formatting
        appendMessage('Your wellness report has been successfully generated and downloaded. The report uses color coding to highlight important information.', 'assistant');
        
    } catch (error) {
        console.error('Error exporting PDF:', error);
        // Show error message to user
        appendMessage(`Failed to export PDF: ${error.message}. Please try again.`, 'assistant');
    }
}

// Add this function to the PDF preview button if you've implemented it
function showPdfPreview() {
    // This function would show a preview modal before actually exporting
    // You can implement this if needed, using the CSS and HTML I provided earlier
    
    // For now, we'll just export directly
    exportPDF();
}

// You can add a preview button to your UI if desired
function addPdfPreviewButton() {
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn && !document.getElementById('preview-pdf-btn')) {
        const previewBtn = document.createElement('button');
        previewBtn.id = 'preview-pdf-btn';
        previewBtn.className = 'action-button';
        previewBtn.innerHTML = '<i class="fas fa-eye"></i> Preview';
        previewBtn.onclick = showPdfPreview;
        
        exportBtn.parentNode.insertBefore(previewBtn, exportBtn);
    }
}

// Theme Management
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

// Utility Functions
function showTypingIndicator() {
    if (state.isTyping) return;
    
    state.isTyping = true;
    const indicator = document.createElement('div');
    indicator.className = 'message assistant-message typing';
    indicator.innerHTML = `
        <div class="message-bubble typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    dom.chatMessages.appendChild(indicator);
    scrollToBottom();
}

function hideTypingIndicator() {
    state.isTyping = false;
    const typingIndicator = document.querySelector('.typing');
    if (typingIndicator) {
        typingIndicator.remove();
    }
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

// Health Check
async function checkHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) {
            console.error('Health check failed');
        }
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initializeApp);