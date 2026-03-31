// PhishGuard++ Popup Logic

// ── Backend Configuration ───────────────────────────────────
const BACKEND_URL = "http://localhost:8000";

document.addEventListener('DOMContentLoaded', async () => {
  const statusBadge = document.getElementById('status-badge');
  const verdictText = document.getElementById('verdict-text');
  const riskPercent = document.getElementById('risk-percent');
  const gaugeFill = document.getElementById('gauge-fill');
  const activeTier = document.getElementById('active-tier');
  const explanation = document.getElementById('explanation');
  const scoreSection = document.querySelector('.score-section');
  const consensusVal = document.getElementById('consensus-val');

  // 1. Get current active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  if (!tab || !tab.url || tab.url.startsWith('chrome://')) {
    statusBadge.innerText = 'Inactive';
    verdictText.innerText = 'Protected Page';
    explanation.innerText = 'PhishGuard++ is active but does not scan internal browser pages for privacy.';
    return;
  }

  // 1b. Fetch Community Consensus immediately from FastAPI
  try {
    const dbResp = await fetch(`${BACKEND_URL}/community/check?url=${encodeURIComponent(tab.url)}`);
    if (dbResp.ok) {
      const data = await dbResp.json();
      if (data && data.found && data.report_count > 0) {
        consensusVal.innerHTML = `🚨 ${data.report_count} Users Reported`;
        consensusVal.style.color = '#ef4444';
      } else {
        consensusVal.innerText = 'Safe (No Reports)';
        consensusVal.style.color = '#10b981';
      }
    }
  } catch(e) {
    console.warn("Community Layer offline or misconfigured");
    consensusVal.innerText = 'Trust Network Unavailable';
  }

  // 2. Request DOM features from content script
  chrome.tabs.sendMessage(tab.id, { action: 'extract_features' }, async (response) => {
    if (chrome.runtime.lastError || !response) {
      statusBadge.innerText = 'Error';
      verdictText.innerText = 'Connection Lost';
      explanation.innerText = 'Please refresh the page to enable real-time protection.';
      return;
    }

    const { url, features, excerpt } = response;

    // 3. Multimodal: Capture visual structure
    let screenshotBase64 = null;
    try {
      screenshotBase64 = await chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 50 });
    } catch (e) {
      console.warn("Screenshot capture blocked:", e);
    }

    // 4. Send to background for 3-tier analysis
    chrome.runtime.sendMessage({ action: 'analyze', url, features, excerpt, screenshot: screenshotBase64 }, (result) => {
      if (result) {
        updateUI(result);
      } else {
        console.error('No response from background script.');
      }
    });
  });

  // 4. Attach event listeners
  const detailsBtn = document.getElementById('details-btn');
  const reportBtnTrigger = document.getElementById('report-btn-trigger');
  const reportForm = document.getElementById('report-form');
  const submitReportBtn = document.getElementById('submit-report-btn');
  const reportReason = document.getElementById('report-reason');

  detailsBtn.addEventListener('click', () => {
    alert(`Technical Details:\nVerdict: ${verdictText.innerText}\nScore: ${riskPercent.innerText}\nTier: ${activeTier.innerText}\n\nAnalysis:\n${explanation.innerText}`);
  });

  reportBtnTrigger.addEventListener('click', () => {
    reportForm.classList.remove('hidden');
    reportBtnTrigger.classList.add('hidden');
    reportReason.focus();
  });

  submitReportBtn.addEventListener('click', async () => {
    submitReportBtn.innerText = "Reporting...";
    submitReportBtn.disabled = true;

    try {
      const resp = await fetch(`${BACKEND_URL}/community/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          url: tab.url,
          reason: reportReason.value.trim() 
        })
      });

      if (resp.ok) {
        submitReportBtn.innerText = "Reported! ✅";
        setTimeout(() => {
          window.close();
        }, 1500);
      } else {
        throw new Error("API Failure");
      }
    } catch(e) {
      console.error("Reporting failed", e);
      submitReportBtn.innerText = "Error Reporting";
      submitReportBtn.disabled = false;
    }
  });

  function updateUI(result) {
    const { verdict, score, tier, reason } = result;
    
    const percent = Math.round(score * 100);
    riskPercent.innerText = `${percent}%`;
    
    // SVG Gauge (half circumference ~ 126)
    const offset = 126 - (percent / 100) * 126;
    gaugeFill.style.strokeDasharray = `${126 - offset}, 126`;

    verdictText.innerText = verdict;
    activeTier.innerText = `Tier ${tier} (${getTierName(tier)})`;
    explanation.innerText = reason || getDefaultReason(verdict, tier);

    // Styling & Glows
    scoreSection.classList.remove('safe-glow', 'warn-glow', 'danger-glow');
    
    if (score < 0.35) {
      gaugeFill.style.stroke = '#10b981';
      statusBadge.innerText = 'Secure';
      verdictText.className = 'safe';
      scoreSection.classList.add('safe-glow');
    } else if (score < 0.75) {
      gaugeFill.style.stroke = '#f59e0b';
      statusBadge.innerText = 'Suspicious';
      verdictText.className = 'warn';
      scoreSection.classList.add('warn-glow');
    } else {
      gaugeFill.style.stroke = '#ef4444';
      statusBadge.innerText = 'Danger';
      verdictText.className = 'danger';
      scoreSection.classList.add('danger-glow');
    }
  }

  function getTierName(tier) {
    if (tier === 1) return 'On-Device';
    if (tier === 2) return 'Cloud ML';
    if (tier === 3) return 'Gemini Vision';
    return 'Community';
  }

  function getDefaultReason(verdict, tier) {
    if (verdict === 'SAFE') return 'Tier 1 validation passed. No structural anomalies detected in URL or source.';
    if (verdict === 'PHISH') return 'Critical structural or visual risk detected by multilayer analysis.';
    return 'Ongoing analysis. Exercise caution while interacting with this page.';
  }
});
