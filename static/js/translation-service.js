// translation-service.js
// This file will handle all translation-related functionality

// Language detection and translation using Hugging Face models
class TranslationService {
    constructor() {
        // Default language is English
        this.currentLanguage = localStorage.getItem('preferredLanguage') || 'en';
        this.huggingFaceToken = null; // Optional: Add your Hugging Face token for higher rate limits
        this.supportedLanguages = [
            { code: 'en', name: 'English' },
            { code: 'fr', name: 'Français' },
            { code: 'es', name: 'Español' },
        ];
        this.isAutoDetectEnabled = localStorage.getItem('autoDetectLanguage') !== 'false';
        
        // Initialize translation cache to reduce API calls
        this.translationCache = {};
    }

    // Initialize the language selector in the UI
    initLanguageSelector(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Create language selector dropdown
        const selectorHtml = `
            <div class="language-selector">
                <select id="language-select" class="language-select">
                    ${this.supportedLanguages.map(lang => 
                        `<option value="${lang.code}" ${lang.code === this.currentLanguage ? 'selected' : ''}>
                            ${lang.name}
                        </option>`
                    ).join('')}
                </select>

            </div>
        `;
        
        container.innerHTML = selectorHtml;
        
        // Add event listeners
        document.getElementById('language-select').addEventListener('change', (e) => {
            this.setLanguage(e.target.value);
        });
        
        document.getElementById('auto-detect-language').addEventListener('change', (e) => {
            this.isAutoDetectEnabled = e.target.checked;
            localStorage.setItem('autoDetectLanguage', this.isAutoDetectEnabled);
        });
    }

    // Set the current language
    setLanguage(languageCode) {
        this.currentLanguage = languageCode;
        localStorage.setItem('preferredLanguage', languageCode);
        this.translateInterface();
        
        // Update the language selector if it exists
        const selector = document.getElementById('language-select');
        if (selector) selector.value = languageCode;
        
        return languageCode;
    }

    // Get the current language
    getLanguage() {
        return this.currentLanguage;
    }

    // Detect language of input text
    async detectLanguage(text) {
        if (!text || text.trim().length < 3) return this.currentLanguage;
        
        try {
            // Use Hugging Face language detection model
            const endpoint = 'https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection';
            
            const headers = {
                'Content-Type': 'application/json',
            };
            
            // Add authorization header if token is available
            if (this.huggingFaceToken) {
                headers['Authorization'] = `Bearer ${this.huggingFaceToken}`;
            }
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ inputs: text.slice(0, 100) }) // Just use the first 100 chars for detection
            });
            
            if (!response.ok) {
                console.warn('Language detection failed:', response.statusText);
                return this.currentLanguage;
            }
            
            const data = await response.json();
            
            // Check if we have valid response data
            if (Array.isArray(data) && data[0] && Array.isArray(data[0])) {
                // Get the most likely language
                const sortedLangs = [...data[0]].sort((a, b) => b.score - a.score);
                if (sortedLangs.length > 0) {
                    // Extract the language code from the label
                    let detectedLang = sortedLangs[0].label.toLowerCase();
                    
                    // Map full language names to codes if needed
                    const languageMap = {
                        'english': 'en',
                        'french': 'fr',
                        'spanish': 'es',
                        
                    };
                    
                    detectedLang = languageMap[detectedLang] || detectedLang;
                    
                    // Check if the detected language is in our supported languages
                    if (this.supportedLanguages.some(lang => lang.code === detectedLang)) {
                        console.log(`Detected language: ${detectedLang}`);
                        return detectedLang;
                    }
                }
            }
            
            return this.currentLanguage;
            
        } catch (error) {
            console.error('Error detecting language:', error);
            return this.currentLanguage;
        }
    }

    // Translate text to target language
    async translateText(text, targetLang = null, sourceLang = null) {
        if (!text || text.trim() === '') return text;
        
        // Default to current language if no target specified
        targetLang = targetLang || this.currentLanguage;
        
        // If target language is English or same as source, return original text
        if (targetLang === 'en' && (!sourceLang || sourceLang === 'en')) {
            return text;
        }
        
        // Generate a cache key
        const cacheKey = `${sourceLang || 'auto'}_${targetLang}_${text}`;
        
        // Check cache first
        if (this.translationCache[cacheKey]) {
            return this.translationCache[cacheKey];
        }
        
        try {
            // If source language is not provided, try to detect it
            if (!sourceLang && this.isAutoDetectEnabled) {
                sourceLang = await this.detectLanguage(text);
            } else if (!sourceLang) {
                sourceLang = 'en'; // Default source language is English
            }
            
            // If source and target languages are the same, return original text
            if (sourceLang === targetLang) {
                return text;
            }
            
            // Construct model endpoint for the specific language pair
            const modelEndpoint = `https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-${sourceLang}-${targetLang}`;
            
            const headers = {
                'Content-Type': 'application/json'
            };
            
            // Add authorization header if token is available
            if (this.huggingFaceToken) {
                headers['Authorization'] = `Bearer ${this.huggingFaceToken}`;
            }
            
            const response = await fetch(modelEndpoint, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ inputs: text })
            });
            
            if (!response.ok) {
                console.warn(`Translation failed: ${sourceLang} to ${targetLang}`);
                
                // Try using English as an intermediate language if direct path fails
                if (sourceLang !== 'en' && targetLang !== 'en') {
                    const textInEnglish = await this.translateText(text, 'en', sourceLang);
                    return await this.translateText(textInEnglish, targetLang, 'en');
                }
                
                return text; // Return original text if translation fails
            }
            
            const data = await response.json();
            
            // Extract translated text
            let translatedText = text; // Default to original
            
            if (Array.isArray(data) && data[0] && data[0].translation_text) {
                translatedText = data[0].translation_text;
            } else if (typeof data === 'object' && data.translation_text) {
                translatedText = data.translation_text;
            }
            
            // Cache the result
            this.translationCache[cacheKey] = translatedText;
            
            return translatedText;
            
        } catch (error) {
            console.error('Translation error:', error);
            return text; // Return original text in case of error
        }
    }

    // Process user message - detect language and translate if needed
    async processUserMessage(message) {
        // If auto-detect is disabled, just return the message
        if (!this.isAutoDetectEnabled) {
            return {
                originalText: message,
                language: this.currentLanguage,
                translatedText: message,
                needsTranslation: false
            };
        }
        
        try {
            // Detect language of user message
            const detectedLanguage = await this.detectLanguage(message);
            
            // If message is not in English, translate to English for the backend
            if (detectedLanguage !== 'en') {
                const translatedText = await this.translateText(message, 'en', detectedLanguage);
                
                return {
                    originalText: message,
                    language: detectedLanguage,
                    translatedText: translatedText,
                    needsTranslation: true
                };
            } else {
                return {
                    originalText: message,
                    language: 'en',
                    translatedText: message,
                    needsTranslation: false
                };
            }
        } catch (error) {
            console.error('Error processing user message:', error);
            return {
                originalText: message,
                language: this.currentLanguage,
                translatedText: message,
                needsTranslation: false
            };
        }
    }

    // Process assistant response - translate if needed
    async processAssistantResponse(response, targetLanguage = null) {
        const userLanguage = targetLanguage || this.currentLanguage;
        
        // If user language is English, no translation needed
        if (userLanguage === 'en') {
            return {
                originalText: response,
                translatedText: response,
                needsTranslation: false
            };
        }
        
        try {
            // Translate from English to user's language
            const translatedText = await this.translateText(response, userLanguage, 'en');
            
            return {
                originalText: response,
                translatedText: translatedText,
                needsTranslation: true
            };
        } catch (error) {
            console.error('Error processing assistant response:', error);
            return {
                originalText: response,
                translatedText: response,
                needsTranslation: false
            };
        }
    }

    // Translate the entire interface
    async translateInterface() {
        // This method would translate all static text elements in the UI
        // For a more comprehensive solution, consider using i18next or a similar library
        
        if (this.currentLanguage === 'en') {
            // Reset to default English text
            this.resetInterfaceText();
            return;
        }
        
        try {
            const elementsToTranslate = document.querySelectorAll('[data-i18n]');
            
            for (const element of elementsToTranslate) {
                const key = element.getAttribute('data-i18n');
                if (key && this.interfaceTranslations[this.currentLanguage] && 
                    this.interfaceTranslations[this.currentLanguage][key]) {
                    element.textContent = this.interfaceTranslations[this.currentLanguage][key];
                }
            }
            
            // Also update placeholders
            const inputElements = document.querySelectorAll('[data-i18n-placeholder]');
            for (const element of inputElements) {
                const key = element.getAttribute('data-i18n-placeholder');
                if (key && this.interfaceTranslations[this.currentLanguage] && 
                    this.interfaceTranslations[this.currentLanguage][key]) {
                    element.placeholder = this.interfaceTranslations[this.currentLanguage][key];
                }
            }
            
        } catch (error) {
            console.error('Error translating interface:', error);
        }
    }

    // Reset interface text to English
    resetInterfaceText() {
        const elementsToReset = document.querySelectorAll('[data-i18n]');
        
        for (const element of elementsToReset) {
            const key = element.getAttribute('data-i18n');
            if (key && this.interfaceTranslations['en'] && this.interfaceTranslations['en'][key]) {
                element.textContent = this.interfaceTranslations['en'][key];
            }
        }
        
        // Reset placeholders
        const inputElements = document.querySelectorAll('[data-i18n-placeholder]');
        for (const element of inputElements) {
            const key = element.getAttribute('data-i18n-placeholder');
            if (key && this.interfaceTranslations['en'] && this.interfaceTranslations['en'][key]) {
                element.placeholder = this.interfaceTranslations['en'][key];
            }
        }
    }

    // Interface translations for static text
    // This is a simple solution - for a production app, consider using a proper i18n library
    interfaceTranslations = {
        'en': {
            'title': 'Mental Wellness Assistant',
            'subtitle': 'Here to support you 24/7',
            'stress_analysis': 'Stress Analysis',
            'current_stress': 'Current stress level:',
            'not_analyzed': 'Not analyzed',
            'toggle_theme': 'Toggle Theme',
            'summary': 'Summary',
            'generate_exercise': 'Generate Exercise',
            'export_pdf': 'Export PDF',
            'send': 'Send',
            'type_message': 'Type your message here...',
            'generating': 'Generating a personalized exercise...',
            'pdf_preparing': 'Preparing your PDF export...',
            'first_message':'Hello! I\'m your wellness assistant. How are you feeling today?'
        },
        'fr': {
            'title': 'Assistant de Bien-être Mental',
            'subtitle': 'Là pour vous soutenir 24/7',
            'stress_analysis': 'Analyse de Stress',
            'current_stress': 'Niveau de stress actuel:',
            'not_analyzed': 'Non analysé',
            'toggle_theme': 'Changer de Thème',
            'summary': 'Résumé',
            'generate_exercise': 'Générer un Exercice',
            'export_pdf': 'Exporter en PDF',
            'send': 'Envoyer',
            'type_message': 'Tapez votre message ici...',
            'generating': 'Génération d\'un exercice personnalisé...',
            'pdf_preparing': 'Préparation de votre export PDF...',
            'first_message':'Bonjour ! Je suis votre assistante bien-être. Comment vous sentez-vous aujourd\'hui ?'

        },
        'es': {
            'title': 'Asistente de Bienestar Mental',
            'subtitle': 'Aquí para apoyarte 24/7',
            'stress_analysis': 'Análisis de Estrés',
            'current_stress': 'Nivel de estrés actual:',
            'not_analyzed': 'No analizado',
            'toggle_theme': 'Cambiar Tema',
            'summary': 'Resumen',
            'generate_exercise': 'Generar Ejercicio',
            'export_pdf': 'Exportar PDF',
            'send': 'Enviar',
            'type_message': 'Escribe tu mensaje aquí...',
            'generating': 'Generando un ejercicio personalizado...',
            'pdf_preparing': 'Preparando tu exportación PDF...',
            'first_mesaage':'Hola, soy tu asistente de bienestar. ¿Cómo te sientes hoy?'

        }
        // Add more languages as needed
    };

    // Debug helper
    logTranslationInfo(originalText, translatedText, sourceLang, targetLang) {
        console.log(`Translation: [${sourceLang} → ${targetLang}]`);
        console.log(`Original: ${originalText}`);
        console.log(`Translated: ${translatedText}`);
    }
}

// Create and export a singleton instance
const translationService = new TranslationService();
export default translationService;