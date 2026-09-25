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
    assert caps["tiers"] == ["free", "pro"]
    assert caps["free_model"] == server.FREE_MODEL
    assert caps["pro_default_deep"] == server.PRO_DEFAULT_DEEP
    assert caps["depths"]["deep"] == {"debate_rounds": 3, "risk_rounds": 3}

def test_free_tier_pins_models(monkeypatch):
    c = _client(monkeypatch)
    job = c.post("/analyze", json={"symbol": "NVDA", "tier": "free",
        "quick_think_llm": "evil/override", "deep_think_llm": "evil/override",
        "research_depth": "deep"}).json()["job_id"]
    r = c.get(f"/result/{job}").json()
    assert r["params"]["quick_think_llm"] == server.FREE_MODEL
    assert r["params"]["deep_think_llm"] == server.FREE_MODEL
    assert r["params"]["max_debate_rounds"] == 3

def test_pro_tier_honors_deep_override(monkeypatch):
    c = _client(monkeypatch)
    job = c.post("/analyze", json={"symbol": "NVDA", "tier": "pro",
        "deep_think_llm": "some/custom-model",
        "analysts": ["market", "news"]}).json()["job_id"]
    r = c.get(f"/result/{job}").json()
    assert r["params"]["deep_think_llm"] == "some/custom-model"
    assert r["params"]["quick_think_llm"] == server.FREE_MODEL
    assert r["params"]["analysts"] == ["market", "news"]

def test_pro_defaults_deep_model(monkeypatch):
    c = _client(monkeypatch)
    job = c.post("/analyze", json={"symbol": "NVDA", "tier": "pro"}).json()["job_id"]
    r = c.get(f"/result/{job}").json()
    assert r["params"]["deep_think_llm"] == server.PRO_DEFAULT_DEEP

def test_bad_params_rejected(monkeypatch):
    c = _client(monkeypatch)
    # empty list = default-all (same as omitted)
    job = c.post("/analyze", json={"symbol": "NVDA", "analysts": []}).json()["job_id"]
    assert c.get(f"/result/{job}").json()["params"]["analysts"] == list(server.ANALYSTS)
    assert c.post("/analyze", json={"symbol": "NVDA", "analysts": ["hacker"]}).status_code == 400
    assert c.post("/analyze", json={"symbol": "NVDA", "research_depth": "ultra"}).status_code == 400
    assert c.post("/analyze", json={"symbol": "NVDA", "tier": "enterprise"}).status_code == 400

def test_cancel_unknown_404s(monkeypatch):
    c = _client(monkeypatch)
    assert c.post("/cancel/nope").status_code == 404

def test_cancel_finished_job_keeps_status(monkeypatch):
    c = _client(monkeypatch)
    job = c.post("/analyze", json={"symbol": "NVDA"}).json()["job_id"]
    assert c.post(f"/cancel/{job}").json()["status"] == "done"
    assert c.get(f"/status/{job}").json()["status"] == "done"

def test_progress_snapshot_states():
    p = server._progress_snapshot({"market_report": "M"}, ["market", "news"])
    assert p["agents"]["market"]["state"] == "completed"
    assert p["agents"]["market"]["report"] == "M"
    assert p["agents"]["news"]["state"] == "researching"
    assert p["agents"]["social"]["state"] == "skipped"
    assert "report" not in p["agents"]["news"]
    full = server._progress_snapshot(dict(FAKE_STATE), ["market"])
    assert full["agents"]["debate"]["state"] == "completed"
    assert "bull" in full["agents"]["debate"]["report"]
    assert full["agents"]["research"]["report"] == "bull wins"

def test_streaming_progress_mid_run(monkeypatch):
    import threading, time
    gate = threading.Event()
    gate.clear()

    class FakeStreamingTA:
        def __init__(self, **kw): pass
        def propagate_stream(self, symbol, date, asset_type="stock"):
            yield {"market_report": "M-live"}
            assert gate.wait(timeout=15)
            yield dict(FAKE_STATE)
            return FAKE_STATE, "Buy"

    monkeypatch.setattr(server, "TradingAgentsGraph", FakeStreamingTA)
    c = TestClient(server.app)
    job = c.post("/analyze", json={"symbol": "NVDA", "analysts": ["market", "news"]}).json()["job_id"]
    prog = None
    for _ in range(50):
        s = c.get(f"/status/{job}").json()
        if s.get("progress", {}).get("agents", {}).get("market", {}).get("state") == "completed":
            prog = s["progress"]
            break
        time.sleep(0.1)
    assert prog is not None, "mid-run progress never appeared"
    assert prog["agents"]["market"]["report"] == "M-live"
    assert prog["agents"]["news"]["state"] == "researching"
    assert prog["agents"]["social"]["state"] == "skipped"
    gate.set()
    for _ in range(50):
        if c.get(f"/status/{job}").json()["status"] == "done":
            break
        time.sleep(0.1)
    r = c.get(f"/result/{job}").json()
    assert r["signal"] == "Buy"

def test_timing_callback_accumulates():
    t = server.TimingCallback()
    t.on_llm_start({}, ["p"], run_id="r1")
    t.on_llm_end({}, run_id="r1")
    t.on_llm_start({}, ["p"], run_id="r2")
    t.on_llm_error(ValueError("x"), run_id="r2")
    s = t.snapshot()
    assert s["llm_calls"] == 1 and s["llm_seconds"] >= 0

def test_apply_snapshot_timings():
    job: dict = {}
    p1 = server._apply_snapshot(job, {"market_report": "M"}, ["market", "news"], 100.0)
    assert "seconds" not in p1["timings"]["market"]  # fresh agent: stamps only, no made-up 0s
    assert p1["agents"]["news"]["state"] == "researching"
    assert "seconds" not in p1["timings"].get("news", {})
    p2 = server._apply_snapshot(job, {"market_report": "M", "sentiment_report": "S",
                                      "news_report": "N"}, ["market", "news"], 160.0)
    assert p2["timings"]["news"]["seconds"] == 60.0
    assert p2["agents"]["news"]["state"] == "completed"