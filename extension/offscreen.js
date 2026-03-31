import { extractUrlFeatures } from './lib/extractor.js';

// ORT is loaded globally via offscreen.html <script src="lib/ort.wasm.min.js">

// Setup ORT Environment for Chrome Extension Context
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let session = null;
let initPromise = null;

async function initModel() {
  try {
    session = await ort.InferenceSession.create('/models/phishguard_edge.onnx', {
      executionProviders: ['wasm'],
    });
    console.log('✅ PhishGuard++ Tier 1 (ONNX) Loaded Successfully in Offscreen Document');
  } catch (e) {
    console.error('❌ Failed to load ONNX model in offscreen document:', e);
  }
}

// Start loading model immediately on document creation
initPromise = initModel();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.target === 'offscreen' && request.action === 'analyze') {
    handleAnalyze(request.url, request.features, request.excerpt)
      .then(result => sendResponse(result))
      .catch(err => {
        console.error('Inference error handlers:', err);
        sendResponse({ verdict: 'ERROR', reason: err.message, tier: 1 });
      });
    return true; // Keep message channel open for async execution
  }
});

async function handleAnalyze(url, htmlFeatures, htmlExcerpt) {
  // Wait for the ONNX model to load
  await initPromise;
  
  if (!session) {
    throw new Error('Tier 1 Edge Model not initialized');
  }

  // Extract baseline 20 URL features
  const urlFeatures = extractUrlFeatures(url);

  // Combine to create the needed 40 features
  const combinedFeatures = [...urlFeatures, ...htmlFeatures];

  // Perform inference robustly
  const inputName = session.inputNames[0] || 'float_input';
  const tensor = new ort.Tensor('float32', new Float32Array(combinedFeatures), [1, 40]);
  const feeds = { [inputName]: tensor };

  let results;
  try {
    // Attempt to fetch all outputs (Label + Probability)
    results = await session.run(feeds);
  } catch (err) {
    if (err.message.includes("Can't access output tensor data") || err.message.includes("error code = 1")) {
      console.warn("ZipMap Serialization failed. Falling back to Label extraction only.");
      // In ONNX Web, ZipMaps crash the WASM bridge. Fetch ONLY the primary label output!
      results = await session.run(feeds, [session.outputNames[0]]);
    } else {
      throw err;
    }
  }
  
  // Pluck probability distribution or label
  let score = 0.5;
  const probOutput = session.outputNames.length > 1 ? results[session.outputNames[1]] : null;
  const labelOutput = results[session.outputNames[0]];

  if (probOutput && probOutput.type === 'tensor' && probOutput.data) {
    score = probOutput.data[1] !== undefined ? probOutput.data[1] : probOutput.data[0];
  } else if (probOutput && Array.isArray(probOutput) && probOutput.length > 0) {
    const probMap = probOutput[0];
    if (probMap instanceof Map) {
      score = probMap.get(1) || probMap.get(1n) || probMap.get('1') || 0.5;
    } else {
      score = probMap[1] || probMap['1'] || probMap[1n] || 0.5;
    }
  } else if (labelOutput && labelOutput.data) {
    // We only have the generic label output due to fallback or model shape
    const label = labelOutput.data[0];
    score = (label === 1n || label === 1) ? 0.90 : 0.10;
  }
  
  console.log(`Tier 1 Inference score for ${url} = ${score.toFixed(4)}`);

  return { score };
}
