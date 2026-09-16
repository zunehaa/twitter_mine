/**
 * SENTIX AI — FRONTEND APPLICATION LOGIC
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tweetInput = document.getElementById('tweet-input');
    const charCounter = document.getElementById('char-counter');
    const btnClear = document.getElementById('btn-clear');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnReset = document.getElementById('btn-reset');
    const btnCopyResult = document.getElementById('btn-copy-result');
    const btnClearHistory = document.getElementById('btn-clear-history');
    
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');
    const loadingState = document.getElementById('loading-state');
    const resultPanel = document.getElementById('result-panel');
    
    const resultBadge = document.getElementById('result-badge');
    const badgeIcon = document.getElementById('badge-icon');
    const badgeText = document.getElementById('badge-text');
    const confidenceVal = document.getElementById('confidence-val');
    const distributionBars = document.getElementById('distribution-bars');
    const detailCleanedText = document.getElementById('detail-cleaned-text');
    const detailModelName = document.getElementById('detail-model-name');
    const explanationText = document.getElementById('explanation-text');
    const modelStatusText = document.getElementById('model-status-text');

    // Dashboard Stats
    const statTotal = document.getElementById('stat-total');
    const statPos = document.getElementById('stat-pos');
    const statNeu = document.getElementById('stat-neu');
    const statNeg = document.getElementById('stat-neg');
    const statIrr = document.getElementById('stat-irr');
    const historyTbody = document.getElementById('history-tbody');

    // Sample Pills
    const samplePills = document.querySelectorAll('.pill-sample');

    // State Variables
    let sessionHistory = JSON.parse(localStorage.getItem('sentix_history') || '[]');
    let currentPrediction = null;

    // Initialize Page
    initHealthCheck();
    renderHistory();
    updateDashboardStats();

    // Event Listeners
    if (tweetInput) {
        tweetInput.addEventListener('input', () => {
            const len = tweetInput.value.length;
            charCounter.textContent = `${len} / 280`;
            hideError();
        });
    }

    if (btnClear) {
        btnClear.addEventListener('click', () => {
            tweetInput.value = '';
            charCounter.textContent = '0 / 280';
            hideError();
            hideResult();
            tweetInput.focus();
        });
    }

    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', analyzeSentiment);
    }

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            tweetInput.value = '';
            charCounter.textContent = '0 / 280';
            hideResult();
            hideError();
            tweetInput.focus();
        });
    }

    if (btnCopyResult) {
        btnCopyResult.addEventListener('click', copyResultToClipboard);
    }

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', clearHistory);
    }

    // Sample Pills Click Handlers
    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleText = pill.getAttribute('data-sample');
            if (sampleText && tweetInput) {
                tweetInput.value = sampleText;
                charCounter.textContent = `${sampleText.length} / 280`;
                hideError();
                hideResult();
                tweetInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                tweetInput.focus();
            }
        });
    });

    // Mobile Navigation Toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.getElementById('nav-links');
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }

    /**
     * Initial Backend Health Check & Metadata
     */
    async function initHealthCheck() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'online') {
                    modelStatusText.textContent = `Model Ready (${data.accuracy}% Acc)`;
                } else {
                    modelStatusText.textContent = 'Model Offline';
                }
            } else {
                modelStatusText.textContent = 'Backend Error';
            }
        } catch (err) {
            console.warn('Backend connection warning:', err);
            modelStatusText.textContent = 'Disconnected';
        }

        fetchModelMetadata();
    }

    async function fetchModelMetadata() {
        try {
            const res = await fetch('/api/model-info');
            if (res.ok) {
                const meta = await res.json();
                const infoType = document.getElementById('info-model-type');
                const infoDataset = document.getElementById('info-dataset-name');
                const infoSamples = document.getElementById('info-samples');

                if (infoType && meta.model_type) infoType.textContent = meta.model_type;
                if (infoDataset && meta.dataset_name) infoDataset.textContent = meta.dataset_name;
                if (infoSamples && meta.total_samples) infoSamples.textContent = `${meta.total_samples.toLocaleString()} training samples`;
            }
        } catch (err) {
            console.log('Metadata fetch deferred');
        }
    }

    /**
     * Main Sentiment Analysis Request
     */
    async function analyzeSentiment() {
        const text = tweetInput ? tweetInput.value.trim() : '';

        if (!text) {
            showError('Please enter a Twitter comment to analyze.');
            tweetInput.focus();
            return;
        }

        hideError();
        hideResult();
        showLoading();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            hideLoading();

            if (!response.ok || data.error) {
                showError(data.error || 'An error occurred during model prediction.');
                return;
            }

            currentPrediction = data;
            displayResult(data);
            saveToHistory(data);
            updateDashboardStats();

        } catch (err) {
            hideLoading();
            showError('The AI model backend is currently unavailable. Please ensure the server is running.');
            console.error('API call failed:', err);
        }
    }

    /**
     * Display Prediction Results
     */
    function displayResult(data) {
        const pred = (data.prediction || 'Neutral').toLowerCase();
        
        // Update Badge
        resultBadge.className = `sentiment-result-badge ${pred}`;
        badgeText.textContent = data.prediction.toUpperCase();

        const icons = {
            positive: '🟢',
            negative: '🔴',
            neutral: '⚪',
            irrelevant: '🟣'
        };
        badgeIcon.textContent = icons[pred] || '⚪';

        // Confidence
        confidenceVal.textContent = `${data.confidence}%`;

        // Cleaned Text & Details
        detailCleanedText.textContent = `"${data.cleaned_text}"`;
        explanationText.textContent = `The trained machine learning classifier predicted '${data.prediction}' with ${data.confidence}% confidence score based on the extracted TF-IDF textual features.`;

        // Render Probability Bars
        renderDistributionBars(data.probabilities, pred);

        // Show Result Panel
        resultPanel.classList.remove('hidden');
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Render Probability Distribution Progress Bars
     */
    function renderDistributionBars(probs, activePred) {
        if (!distributionBars || !probs) return;
        distributionBars.innerHTML = '';

        // Sort probabilities descending
        const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        sortedEntries.forEach(([cls, pct]) => {
            const clsKey = cls.toLowerCase().substring(0, 3);
            const isTop = cls.toLowerCase() === activePred;

            const item = document.createElement('div');
            item.className = 'dist-bar-item';
            item.innerHTML = `
                <div class="dist-bar-info">
                    <span>${cls} ${isTop ? '★' : ''}</span>
                    <span>${pct}%</span>
                </div>
                <div class="dist-bar-track">
                    <div class="dist-bar-fill ${clsKey}" style="width: ${pct}%"></div>
                </div>
            `;
            distributionBars.appendChild(item);
        });
    }

    /**
     * Helper UI functions
     */
    function showError(msg) {
        if (errorMessage) errorMessage.textContent = msg;
        if (errorBanner) errorBanner.classList.remove('hidden');
    }

    function hideError() {
        if (errorBanner) errorBanner.classList.add('hidden');
    }

    function showLoading() {
        if (loadingState) loadingState.classList.remove('hidden');
    }

    function hideLoading() {
        if (loadingState) loadingState.classList.add('hidden');
    }

    function hideResult() {
        if (resultPanel) resultPanel.classList.add('hidden');
    }

    /**
     * Copy Result to Clipboard
     */
    function copyResultToClipboard() {
        if (!currentPrediction) return;
        const textToCopy = `Sentix AI Analysis:\nComment: "${currentPrediction.raw_text}"\nSentiment: ${currentPrediction.prediction}\nConfidence: ${currentPrediction.confidence}%`;

        navigator.clipboard.writeText(textToCopy).then(() => {
            const originalText = btnCopyResult.querySelector('span').textContent;
            btnCopyResult.querySelector('span').textContent = 'Copied!';
            setTimeout(() => {
                btnCopyResult.querySelector('span').textContent = originalText;
            }, 2000);
        });
    }

    /**
     * Save Prediction to History
     */
    function saveToHistory(data) {
        const item = {
            id: Date.now(),
            text: data.raw_text,
            prediction: data.prediction,
            confidence: data.confidence,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };

        sessionHistory.unshift(item);
        if (sessionHistory.length > 20) sessionHistory.pop(); // Keep max 20

        localStorage.setItem('sentix_history', JSON.stringify(sessionHistory));
        renderHistory();
    }

    /**
     * Render Recent History Table
     */
    function renderHistory() {
        if (!historyTbody) return;
        historyTbody.innerHTML = '';

        if (sessionHistory.length === 0) {
            historyTbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="4" class="text-center">No comments analyzed in this session yet. Try analyzing a tweet above!</td>
                </tr>
            `;
            return;
        }

        sessionHistory.forEach(item => {
            const tr = document.createElement('tr');
            const predKey = item.prediction.toLowerCase();

            const truncatedText = item.text.length > 50 ? item.text.substring(0, 48) + '...' : item.text;

            tr.innerHTML = `
                <td>"${escapeHtml(truncatedText)}"</td>
                <td><span class="chip ${predKey.substring(0, 3)}">${item.prediction}</span></td>
                <td><strong>${item.confidence}%</strong></td>
                <td>${item.time}</td>
            `;
            historyTbody.appendChild(tr);
        });
    }

    /**
     * Clear Local History
     */
    function clearHistory() {
        sessionHistory = [];
        localStorage.removeItem('sentix_history');
        renderHistory();
        updateDashboardStats();
    }

    /**
     * Update Dashboard Statistics
     */
    function updateDashboardStats() {
        const total = sessionHistory.length;
        let pos = 0, neu = 0, neg = 0, irr = 0;

        sessionHistory.forEach(item => {
            const p = item.prediction.toLowerCase();
            if (p === 'positive') pos++;
            else if (p === 'negative') neg++;
            else if (p === 'neutral') neu++;
            else if (p === 'irrelevant') irr++;
        });

        if (statTotal) statTotal.textContent = total;
        if (statPos) statPos.textContent = pos;
        if (statNeu) statNeu.textContent = neu;
        if (statNeg) statNeg.textContent = neg;
        if (statIrr) statIrr.textContent = irr;
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});
