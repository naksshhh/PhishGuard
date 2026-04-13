"""Live Cloud Run smoke test."""
import httpx
import asyncio

BASE = "https://phishguard-backend-957267859324.us-central1.run.app"

async def test():
    print("1. Health check...")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE}/docs")
        print(f"   Status: {r.status_code} {'✅' if r.status_code == 200 else '❌'}")

    print("2. Phishing URL test (cold start may take ~60s)...")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/analyze/cloud", json={
            "url": "https://paypal-account-verify.xyz",
            "htmlExcerpt": "<form action='steal.ru'><input name='password'></form>"
        })
        d = r.json()
        verdict = d.get("verdict", "?")
        score = d.get("score", 0)
        tier = d.get("tier", "?")
        status = "✅" if verdict == "PHISH" else "⚠️"
        print(f"   {status} Verdict: {verdict} | Score: {score:.3f} | Tier: {tier}")

asyncio.run(test())
