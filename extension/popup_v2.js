// PhishGuard++ Popup Logic — Premium UI
console.log("🚀 PhishGuard++ Popup Loaded (v2.0.1)");

// ── Backend Configuration ────────────────────────────────
const BACKEND_URL = "https://phishguard-backend-957267859324.us-central1.run.app";

// Circumference of the ring (r=46): 2π×46 ≈ 289
const RING_CIRCUMFERENCE = 289;

document.addEventListener('DOMContentLoaded', async () => {
  // ── DOM refs ─────────────────────────────────────────────
  const statusBadge  = document.getElementById('status-badge');
  const badgeLabel   = statusBadge ? statusBadge.querySelector('.badge-label') : null;
  const verdictText  = document.getElementById('verdict-text');
  const riskPercent  = document.getElementById('risk-percent');
  const ringFill     = document.getElementById('ring-fill');
  const ringGlow     = document.getElementById('ring-glow');
  const activeTier   = document.getElementById('active-tier');
  const explanation  = document.getElementById('explanation');
  const scoreCard    = document.getElementById('score-card');
  const consensusVal = document.getElementById('consensus-val');
  const intelBadge   = document.getElementById('intel-badge');
  const td1 = document.getElementById('td1');
  const td2 = document.getElementById('td2');
  const td3 = document.getElementById('td3');

  // ── Get active tab ────────────────────────────────────────
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab || !tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
    setBadge('safe', 'Inactive');
    verdictText.textContent = 'Protected Page';
    verdictText.className = 'score-label';
    explanation.textContent = 'PhishGuard++ does not scan internal browser pages to protect your privacy.';
    setRing(0, 'safe');
    return;
  }

  // ── Community consensus (non-blocking) ───────────────────
  fetchCommunity(tab.url);

  // ── DOM feature extraction ───────────────────────────────
  chrome.tabs.sendMessage(tab.id, { action: 'extract_features' }, async (response) => {
    if (chrome.runtime.lastError || !response) {
      setBadge('warn', 'Offline');
      verdictText.textContent = 'No Signal';
      explanation.textContent = 'Please refresh the page and re-open PhishGuard++.';
      setRing(0, 'warn');
      return;
    }

    const { url, features, excerpt } = response;

    // Screenshot for visual analysis
    let screenshot = null;
    try {
      screenshot = await chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 50 });
    } catch (_) {}

    // Send to background for cascade analysis
    chrome.runtime.sendMessage(
      { action: 'analyze', url, features, excerpt, screenshot },
      (result) => { if (result) updateUI(result); }
    );
  });

  // ── Event listeners ───────────────────────────────────────
  document.getElementById('details-btn').addEventListener('click', () => {
    alert(`PhishGuard++ Technical Report\n\nVerdict: ${verdictText.textContent}\nRisk Score: ${riskPercent.textContent}\nTier: ${activeTier.textContent}\n\n${explanation.textContent}`);
  });

  const reportTrigger = document.getElementById('report-btn-trigger');
  const reportForm    = document.getElementById('report-form');
  const submitBtn     = document.getElementById('submit-report-btn');
  const reportReason  = document.getElementById('report-reason');

  reportTrigger.addEventListener('click', () => {
    reportForm.classList.remove('hidden');
    reportTrigger.classList.add('hidden');
    reportReason.focus();
  });

  submitBtn.addEventListener('click', async () => {
    submitBtn.textContent = 'Reporting...';
    submitBtn.disabled = true;
    try {
      const resp = await fetch(`${BACKEND_URL}/community/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: tab.url, reason: reportReason.value.trim() })
      });
      if (resp.ok) {
        submitBtn.textContent = '✅ Reported!';
        setTimeout(() => window.close(), 1500);
      } else {
        throw new Error('Non-2xx');
      }
    } catch {
      submitBtn.textContent = '❌ Failed — retry';
      submitBtn.disabled = false;
    }
  });

  // ── Community fetch ───────────────────────────────────────
  async function fetchCommunity(url) {
    try {
      const r = await fetch(`${BACKEND_URL}/community/check?url=${encodeURIComponent(url)}`);
      if (!r.ok) throw new Error();
      const d = await r.json();
      if (d.found && d.report_count > 0) {
        if (consensusVal) {
          consensusVal.textContent = `🚨 ${d.report_count} user${d.report_count > 1 ? 's' : ''} reported`;
          consensusVal.style.color = '#ef4444';
        }
        if (intelBadge) {
          intelBadge.textContent   = 'Flagged';
          intelBadge.style.background = 'rgba(239,68,68,0.1)';
          intelBadge.style.borderColor = 'rgba(239,68,68,0.3)';
          intelBadge.style.color       = '#ef4444';
        }
      } else {
        if (consensusVal) {
          consensusVal.textContent = 'No community reports';
          consensusVal.style.color = '#10b981';
        }
        if (intelBadge) {
          intelBadge.textContent   = 'Clean';
          intelBadge.style.background  = 'rgba(16,185,129,0.1)';
          intelBadge.style.borderColor = 'rgba(16,185,129,0.3)';
          intelBadge.style.color       = '#10b981';
        }
      }
    } catch {
      if (consensusVal) consensusVal.textContent = 'Unavailable';
      if (intelBadge) intelBadge.textContent   = '—';
    }
  }

  // ── Main UI update ────────────────────────────────────────
  function updateUI({ verdict, score, tier, reason }) {
    const pct    = Math.round((score || 0) * 100);
    const level  = score < 0.35 ? 'safe' : score < 0.75 ? 'warn' : 'danger';
    const labels = { safe: 'Secure', warn: 'Suspicious', danger: 'Danger' };

    // Percent display
    if (riskPercent) riskPercent.textContent = `${pct}%`;

    // Verdict label
    if (verdictText) {
      verdictText.textContent = verdict || labels[level];
      verdictText.className   = `score-label ${level}`;
    }

    // Ring fill
    setRing(pct, level);

    // Tier pill & dots
    if (activeTier) activeTier.textContent = `Tier ${tier} · ${getTierName(tier)}`;
    setTierDots(tier, level);

    // Badge
    setBadge(level, labels[level]);

    // Card glow
    if (scoreCard) {
      scoreCard.classList.remove('safe-glow', 'warn-glow', 'danger-glow');
      scoreCard.classList.add(`${level}-glow`);
    }

    // Explanation
    if (explanation) explanation.textContent = reason || getDefaultReason(verdict, tier);
  }

  function setRing(pct, level) {
    if (!ringFill || !ringGlow) return;
    const colors = { safe: '#10b981', warn: '#f59e0b', danger: '#ef4444' };
    const c      = colors[level] || '#3b82f6';
    const filled = (pct / 100) * RING_CIRCUMFERENCE;
    const dash   = `${filled} ${RING_CIRCUMFERENCE}`;
    ringFill.style.strokeDasharray = dash;
    ringGlow.style.strokeDasharray = dash;
    ringFill.style.stroke = c;
    ringGlow.style.stroke = c;
  }

  function setBadge(level, label) {
    if (!statusBadge || !badgeLabel) return;
    statusBadge.className = `badge badge--${level}`;
    badgeLabel.textContent = label;
  }

  function setTierDots(tier, level) {
    const dots = [td1, td2, td3];
    const cls = `active-${level}`;
    dots.forEach((d, i) => {
      if (!d) return;
      d.className = `tdot tdot-${i+1}`;
      if (i < tier) d.classList.add(cls);
    });
  }

  function getTierName(tier) {
    return ['', 'On-Device', 'Cloud ML', 'Gemini Vision', 'Community'][tier] || 'AI';
  }

  function getDefaultReason(verdict, tier) {
    if (verdict === 'SAFE')  return 'No structural, visual, or behavioral anomalies detected across all AI layers.';
    if (verdict === 'PHISH') return 'Critical risk detected — multilayer AI analysis flagged phishing indicators.';
    return 'Analysis in progress. Exercise caution while interacting with this page.';
  }
});
