// PhishGuard++ Gmail Content Script
// Detects when a user opens an email in Gmail and extracts
// subject, sender, and body for phishing analysis.
// Uses MutationObserver to detect Gmail's dynamic DOM changes.

(function () {
  'use strict';

  // Debounce to avoid spamming on rapid DOM changes
  let analysisTimeout = null;
  let lastAnalyzedEmailId = null;

  // ── DOM Selectors (Gmail Web UI) ─────────────────────────
  const SELECTORS = {
    // Sender display name and email
    senderContainer: '.gD',
    // Subject line
    subject: '.hP',
    // Email body
    body: '.a3s.aiL',
    // Email thread container (to detect when a new email is opened)
    threadView: '.nH.if',
    // Individual message in thread
    message: '.gs',
    // The main content area
    mainArea: '[role="main"]',
  };

  // ── Badge Injection ──────────────────────────────────────

  function createBadge() {
    const badge = document.createElement('div');
    badge.id = 'phishguard-email-badge';
    badge.className = 'pg-email-badge pg-email-badge--scanning';
    badge.innerHTML = `
      <div class="pg-email-badge-inner">
        <span class="pg-email-badge-icon">🛡️</span>
        <span class="pg-email-badge-text">Scanning...</span>
      </div>
    `;
    return badge;
  }

  function updateBadge(badge, result) {
    if (!badge) return;

    const { verdict, score, branch, reason, ai_generated_likelihood } = result;

    // Remove old state classes
    badge.className = 'pg-email-badge';

    let icon, text, stateClass;
    if (verdict === 'PHISH') {
      icon = '🚨';
      text = `Phishing Detected (${Math.round(score * 100)}%)`;
      stateClass = 'pg-email-badge--danger';
    } else if (verdict === 'SUSPICIOUS') {
      icon = '⚠️';
      text = `Suspicious (${Math.round(score * 100)}%)`;
      stateClass = 'pg-email-badge--warn';
    } else if (verdict === 'SAFE') {
      icon = '✅';
      text = 'Verified Safe';
      stateClass = 'pg-email-badge--safe';
    } else {
      icon = '❓';
      text = 'Analysis Error';
      stateClass = 'pg-email-badge--error';
    }

    badge.className = `pg-email-badge ${stateClass}`;

    let aiTag = '';
    if (ai_generated_likelihood != null && ai_generated_likelihood > 0.3) {
      aiTag = `<span class="pg-email-badge-ai">AI: ${Math.round(ai_generated_likelihood * 100)}%</span>`;
    }

    badge.innerHTML = `
      <div class="pg-email-badge-inner">
        <span class="pg-email-badge-icon">${icon}</span>
        <span class="pg-email-badge-text">${text}</span>
        ${aiTag}
      </div>
      <div class="pg-email-badge-detail">${reason || ''}</div>
    `;

    // Click to expand/collapse detail
    const detail = badge.querySelector('.pg-email-badge-detail');
    badge.querySelector('.pg-email-badge-inner').addEventListener('click', () => {
      detail.classList.toggle('pg-email-badge-detail--visible');
    });
  }

  function injectBadge(emailContainer) {
    // Don't duplicate
    if (emailContainer.querySelector('#phishguard-email-badge')) return null;

    const badge = createBadge();
    // Insert after the subject/header area
    const subjectEl = emailContainer.querySelector(SELECTORS.subject);
    if (subjectEl && subjectEl.parentElement) {
      subjectEl.parentElement.insertAdjacentElement('afterend', badge);
    } else {
      emailContainer.prepend(badge);
    }
    return badge;
  }

  // ── Email Data Extraction ────────────────────────────────

  function extractEmailData() {
    const data = {
      subject: '',
      sender_display: '',
      sender_email: '',
      body_text: '',
      reply_to: null,
    };

    // Subject
    const subjectEl = document.querySelector(SELECTORS.subject);
    if (subjectEl) {
      data.subject = subjectEl.textContent.trim();
    }

    // Sender — Gmail stores email in the 'email' attribute of .gD
    const senderEl = document.querySelector(SELECTORS.senderContainer);
    if (senderEl) {
      data.sender_display = senderEl.getAttribute('name') || senderEl.textContent.trim();
      data.sender_email = senderEl.getAttribute('email') || '';
    }

    // Body — get the most recently expanded message body
    const bodyEls = document.querySelectorAll(SELECTORS.body);
    if (bodyEls.length > 0) {
      // Last message in thread is typically the newest
      const lastBody = bodyEls[bodyEls.length - 1];
      data.body_text = lastBody.innerText.trim().substring(0, 5000);
    }

    return data;
  }

  function getEmailFingerprint() {
    // Create a unique ID for this email view to avoid re-analysis
    const subjectEl = document.querySelector(SELECTORS.subject);
    const senderEl = document.querySelector(SELECTORS.senderContainer);
    const subject = subjectEl ? subjectEl.textContent.trim() : '';
    const sender = senderEl ? (senderEl.getAttribute('email') || '') : '';
    return `${sender}::${subject}`;
  }

  // ── Analysis Trigger ─────────────────────────────────────

  async function analyzeCurrentEmail() {
    const fingerprint = getEmailFingerprint();

    // Skip if we already analyzed this exact email
    if (fingerprint === lastAnalyzedEmailId) return;
    if (!fingerprint || fingerprint === '::') return;

    lastAnalyzedEmailId = fingerprint;

    const emailData = extractEmailData();
    if (!emailData.body_text || emailData.body_text.length < 20) return;

    console.log('🛡️ PhishGuard++ Email Analysis:', emailData.subject);

    // Inject or find badge
    const threadView = document.querySelector(SELECTORS.threadView) || document.querySelector(SELECTORS.mainArea);
    let badge = document.querySelector('#phishguard-email-badge');
    if (!badge && threadView) {
      badge = injectBadge(threadView);
    }

    // Send to background for analysis
    try {
      chrome.runtime.sendMessage(
        {
          action: 'analyze_email',
          emailData: {
            subject: emailData.subject,
            sender_display: emailData.sender_display,
            sender_email: emailData.sender_email,
            body_text: emailData.body_text,
            reply_to: emailData.reply_to,
          },
        },
        (result) => {
          if (chrome.runtime.lastError) {
            console.warn('PhishGuard++ email analysis failed:', chrome.runtime.lastError);
            return;
          }
          if (result) {
            console.log('🛡️ PhishGuard++ Email Result:', result);
            updateBadge(badge, result);
          }
        }
      );
    } catch (e) {
      console.warn('PhishGuard++ email analysis error:', e);
    }
  }

  // ── MutationObserver ─────────────────────────────────────
  // Gmail dynamically loads email content, so we watch for DOM changes

  function onDomChange() {
    // Debounce: wait 800ms after last DOM change before analyzing
    if (analysisTimeout) clearTimeout(analysisTimeout);

    analysisTimeout = setTimeout(() => {
      // Only analyze if we can see an email body
      const bodyEl = document.querySelector(SELECTORS.body);
      if (bodyEl) {
        analyzeCurrentEmail();
      }
    }, 800);
  }

  // Start observing once the page loads
  function init() {
    console.log('🛡️ PhishGuard++ Gmail Protection Active');

    const observer = new MutationObserver(onDomChange);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // Also check immediately in case an email is already open
    setTimeout(() => {
      const bodyEl = document.querySelector(SELECTORS.body);
      if (bodyEl) analyzeCurrentEmail();
    }, 1500);
  }

  // Wait for Gmail to finish loading
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
