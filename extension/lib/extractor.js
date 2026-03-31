// PhishGuard++ — Feature Extraction Logic (ESM)
// Sync with src/features/url_features.py and html_features.py

const SUSPICIOUS_KEYWORDS = [
  "login", "signin", "verify", "account", "update", "secure",
  "banking", "confirm", "password", "credential", "suspend",
  "alert", "urgent", "unlock", "validate", "authenticate",
  "paypal", "apple", "microsoft", "amazon", "google",
  "facebook", "netflix", "instagram", "whatsapp",
];

const POPULAR_BRANDS = [
  "paypal", "apple", "microsoft", "amazon", "google", "facebook",
  "netflix", "instagram", "twitter", "linkedin", "dropbox", "adobe",
  "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
  "dhl", "fedex", "ups", "usps", "sbi", "icici", "hdfc", "irctc",
];

const KNOWN_TLDS = ["com", "net", "org", "edu", "gov", "info", "biz", "co", "io", "me", "in", "uk", "us", "ca", "au", "de", "fr"];

function shannonEntropy(s) {
  if (!s) return 0;
  const freq = {};
  for (const c of s) freq[c] = (freq[c] || 0) + 1;
  const len = s.length;
  return -Object.values(freq).reduce((acc, count) => {
    const p = count / len;
    return acc + p * Math.log2(p);
  }, 0);
}

export function extractUrlFeatures(url) {
  const features = new Array(20).fill(0);
  let parsed;
  try {
    parsed = new URL(url.includes('://') ? url : 'https://' + url);
  } catch (e) {
    return features;
  }

  const fullDomain = parsed.hostname;
  const path = parsed.pathname;
  const query = parsed.search;

  features[0] = url.length;
  features[1] = fullDomain.length;
  const parts = fullDomain.split('.');
  features[2] = Math.max(0, parts.length - 2);
  features[3] = (url.match(/\d/g) || []).length / Math.max(url.length, 1);
  features[4] = (url.match(/[@!#$%^&*()_+=~`|\\{}[\]<>?]/g) || []).length;
  features[5] = shannonEntropy(fullDomain);
  features[6] = /^(\d{1,3}\.){3}\d{1,3}$/.test(fullDomain) ? 1 : 0;
  features[7] = SUSPICIOUS_KEYWORDS.filter(kw => url.toLowerCase().includes(kw)).length;
  const subdomain = parts.slice(0, -2).join('.');
  features[8] = KNOWN_TLDS.some(tld => subdomain.includes('.' + tld + '.') || subdomain.startsWith(tld + '.')) ? 1 : 0;
  features[9] = path.toLowerCase().includes('paypal') || POPULAR_BRANDS.some(brand => path.toLowerCase().includes(brand)) ? 1 : 0;
  features[10] = url.toLowerCase().includes('xn--') ? 1 : 0;
  features[11] = Math.max(0, (url.match(/\/\//g) || []).length - 1);
  features[12] = parsed.protocol === 'https:' ? 1 : 0;
  const pathSegments = path.split('/').filter(s => s.length > 0);
  features[13] = pathSegments.length;
  features[14] = pathSegments.reduce((max, s) => Math.max(max, s.length), 0);
  features[15] = pathSegments.length > 0 ? pathSegments.reduce((sum, s) => sum + s.length, 0) / pathSegments.length : 0;
  features[16] = (query.match(/\d/g) || []).length;
  features[17] = (url.match(/\./g) || []).length;
  features[18] = (url.match(/\//g) || []).length;
  features[19] = (url.match(/&/g) || []).length;

  return features;
}
