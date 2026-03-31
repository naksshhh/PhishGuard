import firebase_admin
from firebase_admin import credentials, firestore
import logging
from pathlib import Path
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

# Singleton database reference
db = None

def init_firebase():
    """Initialize the Firebase Admin SDK."""
    global db
    if not firebase_admin._apps:
        try:
            cred_path = Path(__file__).parent / "serviceAccountKey.json"
            if not cred_path.exists():
                logger.error(f"Firebase Credentials not found at {cred_path}")
                return False
                
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            logger.info("Successfully connected to Firebase Firestore!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False
    return True

def get_url_doc_id(url: str) -> str:
    """Safely encode a URL to be used as a Firestore document ID."""
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

def report_malicious_url(url: str, reason: str = ""):
    """Report a malicious URL, incrementing count and logging reasons."""
    if not db:
        logger.warning("Firebase not initialized. Cannot record report.")
        return False
        
    doc_id = get_url_doc_id(url)
    doc_ref = db.collection("reported_urls").document(doc_id)
    doc = doc_ref.get()
    
    timestamp = datetime.utcnow()
    
    try:
        if doc.exists:
            data = doc.to_dict()
            reasons = data.get("reasons", [])
            # Append reason uniquely
            if reason and reason not in reasons:
                reasons.append(reason)
                
            doc_ref.update({
                "report_count": firestore.Increment(1),
                "last_reported": timestamp,
                "reasons": reasons
            })
        else:
            doc_ref.set({
                "url": url,
                "report_count": 1,
                "first_reported": timestamp,
                "last_reported": timestamp,
                "reasons": [reason] if reason else [],
                "verdict": "PENDING"
            })
        return True
    except Exception as e:
        logger.error(f"Failed to upsert report for {url}: {e}")
        return False

def get_community_trust(url: str) -> dict:
    """Fetch the community trust intelligence for a specific URL."""
    if not db:
        return {"found": False, "report_count": 0, "error": "DB_NOT_INITIALIZED"}
        
    try:
        doc_id = get_url_doc_id(url)
        doc = db.collection("reported_urls").document(doc_id).get()
        
        if doc.exists:
            data = doc.to_dict()
            return {
                "found": True,
                "report_count": data.get("report_count", 0),
                "reasons": data.get("reasons", []),
                "verdict": data.get("verdict", "PENDING")
            }
        
        return {
            "found": False,
            "report_count": 0
        }
    except Exception as e:
        logger.error(f"Failed to fetch community trust for {url}: {e}")
        return {"found": False, "report_count": 0, "error": str(e)}
