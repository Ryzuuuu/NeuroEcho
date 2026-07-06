const API_URL = 'http://127.0.0.1:8000';

// DOM Elements
const screens = {
    intro: document.getElementById('intro-screen'),
    recording: document.getElementById('recording-screen'),
    loading: document.getElementById('loading-screen'),
    result: document.getElementById('result-screen')
};

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');
const countdownEl = document.getElementById('countdown');
const progressCircle = document.querySelector('.progress-ring__circle');
const riskCircle = document.getElementById('risk-circle');
const probText = document.getElementById('probability-text');
const riskLabel = document.getElementById('risk-label');
const riskMessage = document.getElementById('risk-message');
const actionContainer = document.getElementById('action-container');

// State
let mediaRecorder;
let audioChunks = [];
let recordingTimer;
let timeLeft = 90;
const TOTAL_TIME = 90;

// Initialize Progress Ring
const radius = progressCircle.r.baseVal.value;
const circumference = radius * 2 * Math.PI;
progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
progressCircle.style.strokeDashoffset = 0;

// Initialize Risk Ring
const riskRadius = riskCircle.r.baseVal.value;
const riskCircumference = riskRadius * 2 * Math.PI;
riskCircle.style.strokeDasharray = `${riskCircumference} ${riskCircumference}`;
riskCircle.style.strokeDashoffset = riskCircumference; // Hidden initially

function setProgress(percent) {
    const offset = circumference - percent / 100 * circumference;
    progressCircle.style.strokeDashoffset = offset;
}

function showScreen(screenName) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[screenName].classList.add('active');
}

// Recording Logic
let audioContext;
let processor;
let input;
let globalStream;
let pcmData = [];
let actualSampleRate = 16000;

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        globalStream = stream;
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        actualSampleRate = audioContext.sampleRate; // Get REAL sample rate (browser may ignore our request)
        console.log('[NeuroEcho] AudioContext actual sample rate:', actualSampleRate, 'Hz');
        input = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        pcmData = [];
        
        processor.onaudioprocess = (e) => {
            if (mediaRecorder === 'recording') {
                const channelData = e.inputBuffer.getChannelData(0);
                pcmData.push(new Float32Array(channelData));
            }
        };
        
        input.connect(processor);
        processor.connect(audioContext.destination);
        
        mediaRecorder = 'recording'; // Use string state since we replaced MediaRecorder
        startTimer();
        showScreen('recording');
        
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone. Please ensure permissions are granted.');
    }
}

function stopRecording() {
    if (mediaRecorder === 'recording') {
        mediaRecorder = 'stopped';
        clearInterval(recordingTimer);
        
        if (globalStream) {
            globalStream.getTracks().forEach(track => track.stop());
        }
        if (processor) {
            processor.disconnect();
            input.disconnect();
        }
        if (audioContext) {
            audioContext.close();
        }
        
        showScreen('loading');
        
        // Merge PCM data and encode to WAV
        const length = pcmData.reduce((acc, arr) => acc + arr.length, 0);
        const mergedData = new Float32Array(length);
        let offset = 0;
        for (let arr of pcmData) {
            mergedData.set(arr, offset);
            offset += arr.length;
        }
        
        console.log('[NeuroEcho] PCM samples collected:', mergedData.length, 'at', actualSampleRate, 'Hz');
        console.log('[NeuroEcho] Recording duration:', (mergedData.length / actualSampleRate).toFixed(2), 'seconds');
        const wavBlob = encodeWAV(mergedData, actualSampleRate);
        sendToBackend(wavBlob, 'wav');
    }
}

function encodeWAV(samples, sampleRate) {
    let buffer = new ArrayBuffer(44 + samples.length * 2);
    let view = new DataView(buffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

function startTimer() {
    timeLeft = TOTAL_TIME;
    countdownEl.textContent = timeLeft;
    setProgress(100);

    recordingTimer = setInterval(() => {
        timeLeft--;
        countdownEl.textContent = timeLeft;
        setProgress((timeLeft / TOTAL_TIME) * 100);

        if (timeLeft <= 0) {
            stopRecording();
        }
    }, 1000);
}

// API Interaction
async function sendToBackend(audioBlob, ext) {
    const formData = new FormData();
    // Append the blob with a filename containing the correct extension
    formData.append('file', audioBlob, `recording.${ext}`);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const result = await response.json();
        displayResults(result);
    } catch (err) {
        console.error(err);
        alert('Error communicating with the server. Please check if the backend is running.');
        showScreen('intro');
    }
}

function displayResults(data) {
    showScreen('result');
    
    // Update Risk Ring
    const probPercent = Math.round(data.probability * 100);
    probText.textContent = `${probPercent}%`;
    
    // Animate ring
    setTimeout(() => {
        const offset = riskCircumference - (data.probability * riskCircumference);
        riskCircle.style.strokeDashoffset = offset;
    }, 100);

    // Styling based on band
    let color = '';
    let label = '';
    
    if (data.band === 'GREEN') {
        color = 'var(--green)';
        label = 'Low Risk';
        actionContainer.style.display = 'none';
    } else if (data.band === 'YELLOW') {
        color = 'var(--yellow)';
        label = 'Moderate Risk';
        actionContainer.style.display = 'none';
    } else {
        color = 'var(--red)';
        label = 'High Risk';
        actionContainer.style.display = 'block';
    }

    riskCircle.style.stroke = color;
    probText.style.color = color;
    riskLabel.style.color = color;
    riskLabel.textContent = label;
    riskMessage.textContent = data.message;

    // Update Biomarkers
    if (data.biomarkers) {
        document.getElementById('stat-jitter').textContent = data.biomarkers.jitter_pct.toFixed(2) + '%';
        document.getElementById('stat-shimmer').textContent = data.biomarkers.shimmer_pct.toFixed(2) + '%';
        document.getElementById('stat-hnr').textContent = data.biomarkers.hnr_db.toFixed(1) + 'dB';
    }
}

// Event Listeners
startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
resetBtn.addEventListener('click', () => {
    riskCircle.style.strokeDashoffset = riskCircumference;
    showScreen('intro');
});
