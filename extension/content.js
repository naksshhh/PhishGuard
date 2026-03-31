// PhishGuard++ Content Script
// Extracts 20 structural features from the active tab

const SOCIAL_MEDIA_DOMAINS = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "tiktok.com"];
const AD_DOMAINS = ["doubleclick.net", "googlesyndication.com", "googleadservices.com", "google-analytics.com"];

function getDomain(url) {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch (e) {
    return "";
  }
}

function isExternal(href, originDomain) {
  if (!href || href.startsWith("/") || href.startsWith("#") || href.startsWith("?")) return false;
  if (href.startsWith("javascript:") || href.startsWith("mailto:")) return false;
  const linkDomain = getDomain(href);
  if (!linkDomain) return false;
  return !linkDomain.includes(originDomain) && !originDomain.includes(linkDomain);
}

function extractHtmlFeatures() {
  console.log('PhishGuard++: Extracting HTML features...');
  const originDomain = window.location.hostname.toLowerCase();
  const soup = document; 
  const features = new Array(20).fill(0);

  try {
    const forms = Array.from(soup.forms);
    features[0] = forms.some(f => isExternal(f.action, originDomain)) ? 1 : 0;
    features[1] = soup.querySelectorAll('iframe').length;
    features[2] = soup.querySelectorAll('input[type="hidden"]').length;
    const links = Array.from(soup.querySelectorAll('a[href]'));
    if (links.length > 0) {
      features[3] = links.filter(a => isExternal(a.getAttribute('href'), originDomain)).length / links.length;
    }
    features[4] = soup.querySelectorAll('script').length;
    features[5] = !!soup.querySelector('meta[http-equiv="refresh"]') ? 1 : 0;
    const passwords = soup.querySelectorAll('input[type="password"]');
    features[6] = passwords.length > 0 ? 1 : 0;
    features[7] = passwords.length;
    const title = (soup.title || "").toLowerCase();
    const domainCore = originDomain.split('.')[0];
    features[8] = domainCore && !title.includes(domainCore) ? 1 : 0;
    const icon = soup.querySelector('link[rel*="icon"]');
    features[9] = icon && isExternal(icon.getAttribute('href'), originDomain) ? 1 : 0;
    features[10] = links.filter(a => (a.getAttribute('href') || "").toLowerCase().startsWith('javascript:')).length;
    features[11] = /(?:©|copyright)\s*\d{4}/i.test(soup.body.innerText) ? 1 : 0;
    features[12] = links.filter(a => SOCIAL_MEDIA_DOMAINS.some(sm => (a.getAttribute('href') || "").toLowerCase().includes(sm))).length;
    features[13] = document.documentElement.innerHTML.length;
    const html = document.documentElement.innerHTML;
    features[14] = (html.match(/\s/g) || []).length / Math.max(html.length, 1);
    features[15] = soup.querySelectorAll('noscript').length;
    features[16] = links.filter(a => !isExternal(a.getAttribute('href'), originDomain)).length;
    features[17] = soup.querySelectorAll('img').length;
    features[18] = Array.from(soup.querySelectorAll('script, iframe, a, img')).filter(tag => {
      const src = tag.src || tag.href || "";
      return AD_DOMAINS.some(ad => src.includes(ad));
    }).length;
    features[19] = links.some(a => {
      const text = a.innerText.toLowerCase();
      const href = (a.getAttribute('href') || "").toLowerCase();
      return text.includes('contact') || href.includes('contact');
    }) ? 1 : 0;
  } catch (e) {
    console.warn('PhishGuard++: Partial feature extraction failure', e);
  }

  return features;
}

function getHtmlExcerpt() {
  const head = document.head.innerHTML.substring(0, 500);
  const forms = Array.from(document.forms).map(f => f.outerHTML.substring(0, 500)).join('\n');
  return (head + '\n' + forms).substring(0, 2000);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extract_features') {
    const features = extractHtmlFeatures();
    const excerpt = getHtmlExcerpt();
    sendResponse({ features, excerpt, url: window.location.href });
  }
  return true;
});
