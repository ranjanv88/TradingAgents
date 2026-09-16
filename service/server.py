"""TradingAgents shim: on-demand full-firm runs, in-memory jobs only."""
from __future__ import annotations
import threading, uuid
from datetime import date
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

app = FastAPI(title="tradingagents-shim")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

def _build_graph() -> TradingAgentsGraph:
    cfg = DEFAULT_CONFIG.copy()
    cfg["checkpoint_enabled"] = False
    return TradingAgentsGraph(debug=False, config=cfg)

class AnalyzeRequest(BaseModel):
    symbol: str
    trade_date: str | None = None
    asset_type: str | None = None

def _detect_asset(symbol: str) -> str:
    return "crypto" if symbol.upper().endswith("-USD") else "stock"

def _sections(state: dict[str, Any]) -> dict[str, Any]:
    deb = state.get("investment_debate_state", {}) or {}
    risk = state.get("risk_debate_state", {}) or {}
    return {
        "analysts": {
            "market": state.get("market_report", ""),
            "sentiment": state.get("sentiment_report", ""),
            "news": state.get("news_report", ""),
            "fundamentals": state.get("fundamentals_report", ""),
        },
        "debate": {"bull": deb.get("bull_history", ""), "bear": deb.get("bear_history", ""),
                   "history": deb.get("history", ""), "judge": deb.get("judge_decision", "")},
        "trader_plan": state.get("trader_investment_plan", ""),
        "risk": {"aggressive": risk.get("aggressive_history", ""),
                 "conservative": risk.get("conservative_history", ""),
                 "neutral": risk.get("neutral_history", ""),
                 "judge": risk.get("judge_decision", "")},
        "final_decision": state.get("final_trade_decision", ""),
    }

def _run(job_id: str, symbol: str, trade_date: str, asset_type: str) -> None:
    try:
        ta = _build_graph()
        final_state, signal = ta.propagate(symbol, trade_date, asset_type=asset_type)
        with _lock:
            _jobs[job_id].update({"status": "done", "signal": signal,
                                  "sections": _sections(final_state)})
    except Exception as e:  # noqa: BLE001
        with _lock:
            _jobs[job_id].update({"status": "error", "error": str(e)[:500]})

@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    sym = (req.symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")
    tdate = req.trade_date or date.today().isoformat()
    asset = req.asset_type or _detect_asset(sym)
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "running", "symbol": sym}
    threading.Thread(target=_run, args=(job_id, sym, tdate, asset), daemon=True).start()
    return {"job_id": job_id, "status": "running"}

@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return {"job_id": job_id, "status": job["status"], **({"error": job["error"]} if job.get("error") else {})}

@app.get("/result/{job_id}")
def result(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] != "done":
            return {"job_id": job_id, "status": job["status"]}
        return {"job_id": job_id, "status": "done", "signal": job["signal"],
                "sections": job["sections"], "note": "research-only, no orders"}

@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    cfg = DEFAULT_CONFIG
    return {"provider": cfg.get("llm_provider"), "deep_think_llm": cfg.get("deep_think_llm"),
            "quick_think_llm": cfg.get("quick_think_llm"),
            "analysts": ["market", "social", "news", "fundamentals"],
            "note": "research-only, no orders"}