"""Shim service tests: fake TradingAgentsGraph, no network, no LLM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
import server

FAKE_STATE = {
    "company_of_interest": "NVDA", "trade_date": "2026-09-15",
    "market_report": "M", "sentiment_report": "S", "news_report": "N",
    "fundamentals_report": "F",
    "investment_debate_state": {"bull_history": "bull", "bear_history": "bear",
        "history": "h", "current_response": "c", "judge_decision": "bull wins"},
    "trader_investment_plan": "buy plan",
    "risk_debate_state": {"aggressive_history": "a", "conservative_history": "c",
        "neutral_history": "n", "history": "h", "judge_decision": "ok"},
    "final_trade_decision": "**Rating**: Buy\nbecause reasons",
}

class FakeTA:
    def __init__(self, **kw): pass
    def propagate(self, symbol, date, asset_type="stock"):
        return FAKE_STATE, "Buy"

def _client(monkeypatch):
    monkeypatch.setattr(server, "TradingAgentsGraph", FakeTA)
    return TestClient(server.app)

def test_analyze_runs_job_to_done(monkeypatch):
    c = _client(monkeypatch)
    job = c.post("/analyze", json={"symbol": "nvda"}).json()["job_id"]
    assert c.get(f"/status/{job}").json()["status"] == "done"
    r = c.get(f"/result/{job}").json()
    assert r["signal"] == "Buy"
    assert r["sections"]["analysts"]["market"] == "M"
    assert r["sections"]["debate"]["judge"] == "bull wins"
    assert r["sections"]["trader_plan"] == "buy plan"
    assert r["sections"]["risk"]["judge"] == "ok"
    assert "final_decision" in r["sections"]
    assert r["note"] == "research-only, no orders"

def test_unknown_job_404s(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/status/nope").status_code == 404
    assert c.get("/result/nope").status_code == 404

def test_capabilities_reports_models(monkeypatch):
    c = _client(monkeypatch)
    caps = c.get("/capabilities").json()
    assert caps["analysts"] == ["market", "social", "news", "fundamentals"]
    assert "deep_think_llm" in caps and "quick_think_llm" in caps