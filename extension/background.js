// PhishGuard++ Background Service Worker
// Orchestrates the 3-tier cascade architecture
// Version: 2.0.1

// Configuration
const CONFIG = {
  TIER1_THRESHOLD: 0.35, // Score < 0.35 is SAFE
  PHISH_THRESHOLD: 0.75, // Score > 0.75 is PHISH
  BACKEND_URL: 'https://phishguard-backend-957267859324.us-central1.run.app',
};

// Offscreen Document Management
let creating; // A global promise to avoid concurrency issues
async function setupOffscreenDocument(path) {
  // Check all windows controlled by the service worker to see if one 
  // of them is the offscreen document with the given path
  const offscreenUrl = chrome.runtime.getURL(path);
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [offscreenUrl]
  });

  if (existingContexts.length > 0) {
    return;
  }

  // create offscreen document
  if (creating) {
    await creating;
  } else {
    creating = chrome.offscreen.createDocument({
      url: path,
      reasons: ['WORKERS'],
      justification: 'Inference with ONNX Runtime Web'
    });
    await creating;
    creating = null;
  }
}

// ── Threat Interception & Notifications ───────────────────
function showThreatNotification(url, verdict) {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon128.png',
    title: 'PhishGuard++: Blocked Threat',
    message: `Detected ${verdict} attack on ${url}. Site blocked.`,
    priority: 2
  });
}

async function analyzeUrl(url, htmlFeatures, htmlExcerpt, screenshotBase64, tabId = null) {
  console.log(`🚀 Analyzing: ${url}`);

  // Tier 1: Edge (ONNX) via Offscreen Document
  let score = 0.5;
  try {
    await setupOffscreenDocument('offscreen.html');

    const response = await chrome.runtime.sendMessage({
      target: 'offscreen',
      action: 'analyze',
      url: url,
      features: htmlFeatures,
      excerpt: htmlExcerpt
    });

    if (response && response.error) {
      console.error('Tier 1 Inference Failed internally:', response.error);
    } else if (response && response.score !== undefined) {
      score = response.score;
      console.log(`Tier 1 Score (Offscreen): ${score.toFixed(4)}`);
    } else if (response && response.verdict === 'ERROR') {
      console.error('Tier 1 Inference Error:', response.reason);
    }
  } catch (e) {
    console.error('Tier 1 Inference Communication Failed:', e);
  }

  // Tier 1 Verdict
  if (score < CONFIG.TIER1_THRESHOLD) {
    return { verdict: 'SAFE', score, tier: 1 };
  }

  if (score > CONFIG.PHISH_THRESHOLD) {
    const result = { verdict: 'PHISH', score, tier: 1, reason: 'Detected structural phishing indicators.' };
    if (tabId) triggerInterceptor(tabId, result.reason);
    return result;
  }

  // Tier 2: Cloud (Escalation)
  console.log('⚠️ Suspicious: Escalating to Tier 2 Cloud...');
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/analyze/cloud`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, htmlExcerpt: htmlExcerpt, screenshotBase64: screenshotBase64 })
    });
    const cloudResult = await response.json();
    const finalResult = { ...cloudResult, tier: cloudResult.tier || 2 };

    if (finalResult.verdict === 'PHISH' && tabId) {
      triggerInterceptor(tabId, finalResult.reason);
    }

    return finalResult;
  } catch (e) {
    console.warn('Tier 2 Escalation Failed:', e);
    return { verdict: 'SUSPICIOUS', score, tier: 1, reason: 'Site analysis inconclusive (Cloud Offline).' };
  }
}

function triggerInterceptor(tabId, reason) {
  chrome.tabs.sendMessage(tabId, { action: 'show_warning', reason });
  showThreatNotification('Current Site', 'PHISHING');
}

// Lifecycle Management
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "scan_link",
    title: "Scan link with PhishGuard++",
    contexts: ["link"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "scan_link") {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'PhishGuard++: Remote Scan',
      message: `Analyzing link destination: ${info.linkUrl.substring(0, 50)}...`,
      priority: 1
    });
    
    // Remote analysis (No DOM features available for a link not visited)
    analyzeUrl(info.linkUrl, [], "", null, tab.id);
  }
});

// Ensure offscreen document is ready on start
setupOffscreenDocument('offscreen.html').catch(console.error);

// ── Tier 4: Email Analysis ────────────────────────────────
async function analyzeEmail(emailData) {
  console.log(`📧 PhishGuard++ Email Analysis: "${emailData.subject}"`);
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/analyze/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subject: emailData.subject || '',
        sender_display: emailData.sender_display || '',
        sender_email: emailData.sender_email || '',
        body_text: emailData.body_text || '',
        reply_to: emailData.reply_to || null,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const result = await response.json();
    console.log(`📧 Email Result:`, result);
    return result;
  } catch (e) {
    console.warn('Email analysis failed:', e);
    return {
      verdict: 'ERROR',
      score: 0.0,
      branch: 'error',
      reason: `Analysis unavailable: ${e.message}`,
    };
  }
}

// Message Listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyze' && request.target !== 'offscreen') {
    analyzeUrl(request.url, request.features, request.excerpt, request.screenshot, sender.tab?.id)
      .then(result => {
        if (result) sendResponse(result);
        else sendResponse({ verdict: 'ERROR', reason: 'No result from model', tier: 1 });
      })
      .catch(err => sendResponse({ verdict: 'ERROR', reason: err.message, tier: 1 }));
    return true;
  }

  if (request.action === 'analyze_email') {
    analyzeEmail(request.emailData)
      .then(result => sendResponse(result))
      .catch(err => sendResponse({ verdict: 'ERROR', reason: err.message, branch: 'error' }));
    return true;
  }
});
