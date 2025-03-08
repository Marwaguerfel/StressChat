class EmotionDetector {
    constructor() {
        // State variables
        this.isWebcamActive = false;
        this.stream = null;
        this.capturedEmotions = [];
        this.lastEmotionTimestamp = 0;
        this.emotionUpdateInterval = 1000; // Update emotion every 1 second
        this.emotionDetectionActive = false;
        this.currentEmotion = null;
        this.videoCapture = null;
        this.isCameraDenied = false;
        
        // Check for emotion detection availability
        this.checkAvailability();
    }

    // Initialize emotion detection
    init() {
        this.setupDomElements();
        this.setupEventListeners();
    }

    // Check if camera is available
    async checkAvailability() {
        try {
            // Check if MediaDevices API is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn("Emotion detection not available: Camera API not supported");
                this.isCameraDenied = true;
                return false;
            }
            
            // Try to get camera permission
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            // Stop the stream right away, we just wanted to check permission
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.warn("Emotion detection not available:", error.message);
            this.isCameraDenied = true;
            return false;
        }
    }

    // Setup DOM elements
    setupDomElements() {
        this.elements = {
            enableCameraToggle: document.getElementById('enable-camera'),
            emotionButton: document.getElementById('emotion-button'),
            videoContainer: document.getElementById('video-container'),
            webcamVideo: document.getElementById('webcam'),
            captureButton: document.getElementById('capture-emotion'),
            closeButton: document.getElementById('close-webcam'),
            liveEmotion: document.getElementById('live-emotion'),
            emotionIcon: document.getElementById('emotion-icon'),
            emotionText: document.getElementById('emotion-text')
        };
        
        // Hide emotion button if camera is not available
        if (this.isCameraDenied) {
            if (this.elements.emotionButton) {
                this.elements.emotionButton.style.display = 'none';
            }
            if (this.elements.enableCameraToggle) {
                this.elements.enableCameraToggle.disabled = true;
                this.elements.enableCameraToggle.parentElement.title = "Camera access denied or not available";
            }
        }
    }

    // Setup event listeners
    setupEventListeners() {
        // Toggle camera access
        if (this.elements.enableCameraToggle) {
            this.elements.enableCameraToggle.addEventListener('change', (e) => {
                this.toggleContinuousEmotionDetection(e.target.checked);
            });
        }
        
        // Open webcam for a single capture
        if (this.elements.emotionButton) {
            this.elements.emotionButton.addEventListener('click', () => {
                this.startWebcam();
            });
        }
        
        // Capture current emotion
        if (this.elements.captureButton) {
            this.elements.captureButton.addEventListener('click', () => {
                this.captureEmotion();
            });
        }
        
        // Close webcam
        if (this.elements.closeButton) {
            this.elements.closeButton.addEventListener('click', () => {
                this.stopWebcam();
            });
        }
    }

    // Toggle continuous emotion detection
    async toggleContinuousEmotionDetection(enable) {
        if (enable) {
            try {
                // Start continuous emotion detection
                this.emotionDetectionActive = true;
                await this.startBackgroundEmotionDetection();
            } catch (error) {
                console.error("Failed to start emotion detection:", error);
                if (this.elements.enableCameraToggle) {
                    this.elements.enableCameraToggle.checked = false;
                }
                this.emotionDetectionActive = false;
                
                // Show error message
                this.updateEmotionDisplay(null, "Error: " + error.message);
            }
        } else {
            // Stop continuous emotion detection
            this.stopBackgroundEmotionDetection();
            this.emotionDetectionActive = false;
            this.updateEmotionDisplay(null, "Not analyzing");
        }
    }

    // Start webcam for a single capture
    async startWebcam() {
        if (this.isWebcamActive) return;
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'user',
                    width: { ideal: 480 },  // Increased from 320
                    height: { ideal: 360 }  // Increased from 240
                } 
            });
            
            if (this.elements.webcamVideo) {
                this.elements.webcamVideo.srcObject = this.stream;
                this.elements.videoContainer.classList.remove('hidden');
                
                // Wait for video to be ready
                await new Promise(resolve => {
                    this.elements.webcamVideo.onloadedmetadata = () => {
                        this.elements.webcamVideo.play();
                        resolve();
                    };
                });
                
                this.isWebcamActive = true;
                
                // Start periodic emotion detection for live preview
                this.startLiveEmotionPreview();
                
                console.log("Webcam started successfully");
            }
        } catch (error) {
            console.error("Error accessing webcam:", error);
            alert("Could not access camera. Please check your camera permissions.");
        }
    }

    // Stop webcam
    stopWebcam() {
        if (!this.isWebcamActive) return;
        
        // Stop all tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.elements.videoContainer) {
            this.elements.videoContainer.classList.add('hidden');
        }
        
        if (this.elements.webcamVideo) {
            this.elements.webcamVideo.srcObject = null;
        }
        
        this.isWebcamActive = false;
        console.log("Webcam stopped");
    }

    // Start live emotion preview
    startLiveEmotionPreview() {
        // Clear any existing interval
        this.stopLiveEmotionPreview();
        
        // Set up periodic emotion detection
        this.liveEmotionInterval = setInterval(() => {
            this.detectLiveEmotion();
        }, 1000); // Check every second
        
        console.log("Live emotion preview started");
    }

    // Stop live emotion preview
    stopLiveEmotionPreview() {
        if (this.liveEmotionInterval) {
            clearInterval(this.liveEmotionInterval);
            this.liveEmotionInterval = null;
            console.log("Live emotion preview stopped");
        }
    }

    // Detect emotion from live video
    async detectLiveEmotion() {
        if (!this.isWebcamActive || !this.elements.webcamVideo || !this.elements.webcamVideo.srcObject) {
            console.log("Webcam not active or video element not ready");
            return;
        }
        
        try {
            // Make sure video is ready
            if (this.elements.webcamVideo.readyState < 2) {
                console.log("Video not ready yet, state:", this.elements.webcamVideo.readyState);
                return;
            }
            
            // Create a canvas to capture current video frame
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Use larger dimensions for better detection
            canvas.width = this.elements.webcamVideo.videoWidth;
            canvas.height = this.elements.webcamVideo.videoHeight;
            
            console.log(`Canvas dimensions: ${canvas.width}x${canvas.height}, Video dimensions: ${this.elements.webcamVideo.videoWidth}x${this.elements.webcamVideo.videoHeight}`);
            
            // Check if we have valid video dimensions
            if (canvas.width <= 0 || canvas.height <= 0) {
                console.warn("Invalid canvas dimensions, skipping emotion detection");
                return;
            }
            
            // Draw current video frame to canvas
            try {
                context.drawImage(this.elements.webcamVideo, 0, 0, canvas.width, canvas.height);
                console.log("Drew video frame to canvas");
            } catch (drawError) {
                console.error("Error drawing to canvas:", drawError);
                return;
            }
            
            // Convert canvas to base64 data URL with higher quality
            const imageData = canvas.toDataURL('image/jpeg', 0.9); // Higher quality for better detection
            
            console.log(`Image data length: ${imageData.length}, starts with: ${imageData.substring(0, 50)}...`);
            
            // Only detect if we haven't checked recently (to avoid flooding the server)
            const now = Date.now();
            if (now - this.lastEmotionTimestamp > this.emotionUpdateInterval) {
                this.lastEmotionTimestamp = now;
                
                // Send to server for emotion detection
                const response = await fetch('/detect-emotion', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_data: imageData })
                });
                
                console.log("Response status:", response.status);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log("Emotion detection response:", data);
                    
                    if (data.success && data.emotion_analysis) {
                        const emotion = data.emotion_analysis.emotion;
                        const confidence = data.emotion_analysis.confidence;
                        
                        // Update live emotion display
                        if (this.elements.liveEmotion) {
                            this.elements.liveEmotion.textContent = `${emotion} (${(confidence * 100).toFixed(0)}%)`;
                        }
                        this.currentEmotion = data.emotion_analysis;
                        console.log("Detected emotion:", emotion, "with confidence:", confidence);
                    } else {
                        console.warn("No emotion detected, adjusting camera position might help");
                        if (this.elements.liveEmotion) {
                            this.elements.liveEmotion.textContent = `Waiting for face...`;
                        }
                    }
                } else {
                    // Try to get error details
                    let errorDetails = "";
                    try {
                        const errorData = await response.json();
                        errorDetails = errorData.error || response.statusText;
                    } catch (e) {
                        errorDetails = response.statusText;
                    }
                    
                    console.error("Emotion detection request failed:", response.status, errorDetails);
                    if (this.elements.liveEmotion) {
                        this.elements.liveEmotion.textContent = `Server error: ${response.status}`;
                    }
                }
            }
        } catch (error) {
            console.error("Error detecting live emotion:", error);
            if (this.elements.liveEmotion) {
                this.elements.liveEmotion.textContent = "Detection error";
            }
        }
    }

    // Capture the current emotion for use in chat
    captureEmotion() {
        if (!this.currentEmotion) {
            alert("No emotion detected. Please wait for emotion detection.");
            return;
        }
        
        // Store the captured emotion
        this.capturedEmotions.push(this.currentEmotion);
        
        // Update the emotion display
        this.updateEmotionDisplay(this.currentEmotion);
        
        // Close the webcam
        this.stopWebcam();
        
        // Notify the main app that an emotion was captured
        const captureEvent = new CustomEvent('emotion-captured', { 
            detail: { emotion: this.currentEmotion }
        });
        document.dispatchEvent(captureEvent);
        
        console.log("Emotion captured:", this.currentEmotion);
    }

    // Start background emotion detection
    async startBackgroundEmotionDetection() {
        if (!this.emotionDetectionActive) return;
        
        try {
            this.backgroundStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'user',
                    width: { ideal: 320 },  // Larger for better detection
                    height: { ideal: 240 }
                } 
            });
            
            // Create hidden video element for background processing
            this.backgroundVideo = document.createElement('video');
            this.backgroundVideo.srcObject = this.backgroundStream;
            this.backgroundVideo.autoplay = true;
            this.backgroundVideo.style.display = 'none';
            document.body.appendChild(this.backgroundVideo);
            
            // Wait for video to be ready
            await new Promise(resolve => {
                this.backgroundVideo.onloadedmetadata = () => {
                    this.backgroundVideo.play();
                    resolve();
                };
            });
            
            // Start periodic emotion detection
            this.backgroundEmotionInterval = setInterval(() => {
                this.detectBackgroundEmotion();
            }, 5000); // Check every 5 seconds to avoid overloading
            
            // Update display to show that emotion detection is active
            this.updateEmotionDisplay(null, "Detecting...");
            
            console.log("Background emotion detection started");
        } catch (error) {
            console.error("Error starting background emotion detection:", error);
            throw error;
        }
    }

    // Stop background emotion detection
    stopBackgroundEmotionDetection() {
        if (this.backgroundEmotionInterval) {
            clearInterval(this.backgroundEmotionInterval);
            this.backgroundEmotionInterval = null;
        }
        
        if (this.backgroundStream) {
            this.backgroundStream.getTracks().forEach(track => track.stop());
            this.backgroundStream = null;
        }
        
        if (this.backgroundVideo) {
            document.body.removeChild(this.backgroundVideo);
            this.backgroundVideo = null;
        }
        
        console.log("Background emotion detection stopped");
    }

    // Detect emotion from background video
    async detectBackgroundEmotion() {
        if (!this.emotionDetectionActive || !this.backgroundVideo) {
            console.log("Background emotion detection not active or video not available");
            return;
        }
        
        try {
            // Create a canvas to capture current video frame
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Use dimensions from the video
            canvas.width = this.backgroundVideo.videoWidth;
            canvas.height = this.backgroundVideo.videoHeight;
            
            // Check if we have valid video dimensions
            if (canvas.width <= 0 || canvas.height <= 0 || !this.backgroundVideo.videoWidth) {
                console.warn("Invalid background video dimensions, skipping detection");
                return;
            }
            
            // Draw current video frame to canvas
            try {
                context.drawImage(this.backgroundVideo, 0, 0, canvas.width, canvas.height);
            } catch (drawError) {
                console.error("Error drawing background frame to canvas:", drawError);
                return;
            }
            
            // Convert canvas to base64 data URL with better quality
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            
            console.log(`Background image data length: ${imageData.length}`);
            
            // Send to server for emotion detection
            const response = await fetch('/detect-emotion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: imageData })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log("Background emotion detection response:", data);
                
                if (data.success && data.emotion_analysis) {
                    const emotion = data.emotion_analysis;
                    
                    // Update emotion display
                    this.updateEmotionDisplay(emotion);
                    
                    // Store the emotion
                    this.currentEmotion = emotion;
                    this.capturedEmotions.push(emotion);
                    
                    // Notify the main app
                    const updateEvent = new CustomEvent('emotion-updated', { 
                        detail: { emotion: emotion }
                    });
                    document.dispatchEvent(updateEvent);
                    
                    console.log("Background emotion detected:", emotion);
                } else {
                    console.log("No emotion detected in background frame");
                    this.updateEmotionDisplay(null, "Waiting for face...");
                }
            } else {
                console.error("Background emotion detection request failed:", response.status);
            }
        } catch (error) {
            console.error("Error detecting background emotion:", error);
        }
    }

    // Update the emotion display in UI
    updateEmotionDisplay(emotion, customText = null) {
        if (!this.elements.emotionIcon || !this.elements.emotionText) return;
        
        if (emotion) {
            const emotionName = emotion.emotion;
            const confidence = emotion.confidence;
            
            // Update the icon
            this.elements.emotionIcon.innerHTML = `<i class="fas fa-${this.getEmotionIcon(emotionName)}"></i>`;
            
            // Add animation effect
            this.elements.emotionIcon.classList.add('emotion-pulse');
            setTimeout(() => {
                this.elements.emotionIcon.classList.remove('emotion-pulse');
            }, 1000);
            
            // Update the text
            this.elements.emotionText.textContent = `Detected: ${emotionName} (${(confidence * 100).toFixed(0)}%)`;
            
            // Update icon color based on emotion
            this.elements.emotionIcon.style.color = this.getEmotionColor(emotionName);
            
            console.log("Updated emotion display:", emotionName, confidence);
        } else if (customText) {
            // Reset icon to default
            this.elements.emotionIcon.innerHTML = `<i class="fas fa-face-meh"></i>`;
            this.elements.emotionIcon.style.color = '';
            
            // Set custom text
            this.elements.emotionText.textContent = customText;
            
            console.log("Updated emotion display with custom text:", customText);
        }
    }

    // Get Font Awesome icon for emotion
    getEmotionIcon(emotion) {
        const emotionIcons = {
            'happy': 'face-smile',
            'happys': 'face-smile', // Add this line
            'sad': 'face-sad-tear',
            'angry': 'face-angry',
            'neutral': 'face-meh',
            'surprised': 'face-surprise',
            'fearful': 'face-fearful',
            'disgusted': 'face-dizzy',
            // Fallbacks for different model outputs
            'joy': 'face-smile',
            'happiness': 'face-smile',
            'sadness': 'face-sad-tear',
            'anger': 'face-angry',
            'surprise': 'face-surprise',
            'fear': 'face-fearful',
            'disgust': 'face-dizzy',
            'contempt': 'face-rolling-eyes'
        };
        
        return emotionIcons[emotion.toLowerCase()] || 'face-meh';
    }

    // Get color for emotion
    getEmotionColor(emotion) {
        const emotionColors = {
            'happy': '#00e9b8',   // var(--accent-secondary)
            'happys': '#00e9b8',  // Add this line
            'sad': '#3a98f5',     // var(--primary-dark)
            'angry': '#ef4444',   // var(--danger-color)
            'neutral': '#696969', // var(--text-secondary)
            'surprised': '#bb74f0', // var(--accent-tertiary)
            'fearful': '#bb74f0',   // var(--warning-color)
            'disgusted': '#3f3c64', // var(--primary-color)
            // Fallbacks for different model outputs
            'joy': '#00e9b8',
            'happiness': '#00e9b8',
            'sadness': '#3a98f5',
            'anger': '#ef4444',
            'surprise': '#bb74f0',
            'fear': '#bb74f0',
            'disgust': '#3f3c64',
            'contempt': '#3f3c64'
        };
        
        return emotionColors[emotion.toLowerCase()] || '#696969';
    }

    // Get the most recent emotion
    getMostRecentEmotion() {
        if (this.capturedEmotions.length === 0) return null;
        return this.capturedEmotions[this.capturedEmotions.length - 1];
    }

    // Clear captured emotions
    clearEmotions() {
        this.capturedEmotions = [];
        this.updateEmotionDisplay(null, "Not analyzed");
        console.log("Cleared emotions");
    }

    // Get emotion data for use with message sending
    getEmotionData() {
        return this.getMostRecentEmotion();
    }

    // Create emotion badge element
    createEmotionBadge(emotion) {
        if (!emotion) return null;
        
        const badge = document.createElement('span');
        badge.className = `emotion-badge emotion-badge-${emotion.emotion.toLowerCase()}`;
        badge.innerHTML = `<i class="fas fa-${this.getEmotionIcon(emotion.emotion)}"></i> ${emotion.emotion}`;
        
        return badge;
    }
}

export default EmotionDetector;