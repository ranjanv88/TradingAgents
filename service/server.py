"""TradingAgents shim: on-demand full-firm runs, in-memory jobs only."""
from __future__ import annotations
import threading, time, uuid
from datetime import date
from typing import Any
from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

app = FastAPI(title="tradingagents-shim")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

ANALYSTS = ("market", "social", "news", "fundamentals")
DEPTH_ROUNDS = {"shallow": (1, 1), "medium": (2, 2), "deep": (3, 3)}
FREE_MODEL = "nvidia/nemotron-3.5-lightning:free"
PRO_DEFAULT_DEEP = "meta/muse-spark-1.3-contributor"

def _build_graph(overrides: dict[str, Any], analysts: list[str],
                 callbacks: list | None = None) -> TradingAgentsGraph:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: overrides[k] for k in
                ("quick_think_llm", "deep_think_llm",
                 "max_debate_rounds", "max_risk_discuss_rounds")})
    cfg["checkpoint_enabled"] = False
    return TradingAgentsGraph(debug=False, config=cfg,
                              selected_analysts=tuple(analysts),
                              callbacks=callbacks or [])


class TimingCallback(BaseCallbackHandler):
    """Aggregate wall time spent inside LLM calls for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starts: dict[str, float] = {}
        self.llm_seconds = 0.0
        self.llm_calls = 0

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        with self._lock:
            self._starts[str(kwargs.get("run_id"))] = time.monotonic()

    def _finish(self, run_id: Any, ok: bool) -> None:
        with self._lock:
            s = self._starts.pop(str(run_id), None)
            if s is not None and ok:
                self.llm_seconds += time.monotonic() - s
                self.llm_calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._finish(kwargs.get("run_id"), True)

    def on_llm_error(self, error: Any, **kwargs: Any) -> None:
        self._finish(kwargs.get("run_id"), False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"llm_seconds": round(self.llm_seconds, 1),
                    "llm_calls": self.llm_calls}

class AnalyzeRequest(BaseModel):
    symbol: str
    trade_date: str | None = None
    asset_type: str | None = None
    analysts: list[str] | None = None
    research_depth: str | None = None
    tier: str | None = None
    quick_think_llm: str | None = None
    deep_think_llm: str | None = None

def _resolve_job(req: AnalyzeRequest) -> dict[str, Any]:
    analysts = list(req.analysts) if req.analysts else list(ANALYSTS)
    bad = [a for a in analysts if a not in ANALYSTS]
    if not analysts or bad:
        raise HTTPException(
            400, f"analysts must be a non-empty subset of {list(ANALYSTS)}")
    depth = (req.research_depth or "shallow").lower()
    if depth not in DEPTH_ROUNDS:
        raise HTTPException(400, "research_depth must be shallow|medium|deep")
    tier = (req.tier or "free").lower()
    if tier not in ("free", "pro"):
        raise HTTPException(400, "tier must be free|pro")
    if tier == "free":
        # Free reports are pinned server-side; client values ignored.
        quick, deep = FREE_MODEL, FREE_MODEL
    else:
        deep = (req.deep_think_llm or "").strip() or PRO_DEFAULT_DEEP
        quick = (req.quick_think_llm or "").strip() or FREE_MODEL
    debate, risk = DEPTH_ROUNDS[depth]
    return {"analysts": analysts, "depth": depth, "tier": tier,
            "quick_think_llm": quick, "deep_think_llm": deep,
            "max_debate_rounds": debate, "max_risk_discuss_rounds": risk}

def _clip(s: Any, limit: int = 8000) -> str:
    t = s if isinstance(s, str) else ("" if s is None else str(s))
    return t if len(t) <= limit else t[:limit] + "\n…[truncated]"


def _progress_snapshot(merged: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    """Derive per-agent completed/researching/pending/skipped states + reports."""
    sel = [a for a in ANALYSTS if a in (selected or [])] or list(ANALYSTS)
    rep = {
        "market": _clip(merged.get("market_report", "")),
        "social": _clip(merged.get("sentiment_report", "")),
        "news": _clip(merged.get("news_report", "")),
        "fundamentals": _clip(merged.get("fundamentals_report", "")),
        "trader": _clip(merged.get("trader_investment_plan", "")),
        "portfolio": _clip(merged.get("final_trade_decision", "")),
    }
    deb = merged.get("investment_debate_state", {}) or {}
    bull, bear = _clip(deb.get("bull_history", "")), _clip(deb.get("bear_history", ""))
    rjudge = _clip(deb.get("judge_decision", ""))
    risk = merged.get("risk_debate_state", {}) or {}
    kjudge = _clip(risk.get("judge_decision", ""))
    done = {
        "market": bool(rep["market"]), "social": bool(rep["social"]),
        "news": bool(rep["news"]), "fundamentals": bool(rep["fundamentals"]),
        "debate": bool(bull and bear), "research": bool(rjudge),
        "trader": bool(rep["trader"]), "risk": bool(kjudge),
        "portfolio": bool(rep["portfolio"]),
    }
    reports: dict[str, str] = dict(rep)
    reports["debate"] = _clip(f"Bull:\n{bull}\n\nBear:\n{bear}") if done["debate"] else ""
    reports["research"] = rjudge
    reports["risk"] = kjudge
    order = sel + ["debate", "research", "trader", "risk", "portfolio"]
    researching = next((k for k in order if not done.get(k)), None)
    agents: dict[str, Any] = {}
    for k in ANALYSTS:
        if k not in sel:
            agents[k] = {"state": "skipped"}
        else:
            agents[k] = {"state": "completed" if done[k]
                         else ("researching" if researching == k else "pending")}
            if done[k] and rep[k]:
                agents[k]["report"] = rep[k]
    for k in ("debate", "research", "trader", "risk", "portfolio"):
        agents[k] = {"state": "completed" if done[k]
                     else ("researching" if researching == k else "pending")}
        if done[k] and reports[k]:
            agents[k]["report"] = reports[k]
    return {"agents": agents}


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

def _apply_snapshot(job: dict[str, Any], snap: dict[str, Any],
                    analysts: list[str], now: float) -> dict[str, Any]:
    """Merge a stream snapshot into job progress + per-agent timings."""
    prog = _progress_snapshot(snap, analysts)
    times = job.setdefault("agent_times", {})
    for key, a in prog["agents"].items():
        st = a["state"]
        rec = times.setdefault(key, {})
        # fresh = first sighting of this agent as active. When an agent flips
        # pending -> completed inside a single snapshot gap (stream yields are
        # coarser than agent transitions), started_at and completed_at would
        # land on the same tick and fake a "0s" duration. In that case record
        # both stamps but OMIT seconds — the UI shows a plain "completed"
        # rather than a made-up 0s. Real durations come from agents observed
        # in "researching" on an earlier tick (fresh == False here).
        fresh = "started_at" not in rec
        if st in ("researching", "completed") and fresh:
            rec["started_at"] = now
        if st == "completed" and "completed_at" not in rec:
            rec["completed_at"] = now
            if not fresh and "started_at" in rec:
                rec["seconds"] = round(rec["completed_at"] - rec["started_at"], 1)
    prog["timings"] = {k: v for k, v in times.items() if v}
    job["progress"] = prog
    return prog


def _run(job_id: str, symbol: str, trade_date: str, asset_type: str,
         params: dict[str, Any]) -> None:
    timer = TimingCallback()
    try:
        ta = _build_graph(params, params["analysts"], callbacks=[timer])
        streamer = getattr(ta, "propagate_stream", None)
        if streamer is None:
            final_state, signal = ta.propagate(symbol, trade_date, asset_type=asset_type)
        else:
            gen = streamer(symbol, trade_date, asset_type=asset_type)
            while True:
                try:
                    snap = next(gen)
                except StopIteration as e:
                    final_state, signal = e.value
                    break
                with _lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        if job.get("cancel_requested"):
                            job.update({"status": "cancelled"})
                            return
                        _apply_snapshot(job,
                                        snap if isinstance(snap, dict) else {},
                                        params["analysts"], time.time())
                        job["usage"] = timer.snapshot()
        with _lock:
            _jobs[job_id].update({"status": "done", "signal": signal,
                                  "sections": _sections(final_state),
                                  "params": params,
                                  "usage": timer.snapshot()})
    except Exception as e:  # noqa: BLE001
        with _lock:
            _jobs[job_id].update({"status": "error", "error": str(e)[:500],
                                  "usage": timer.snapshot()})

@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    sym = (req.symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")
    params = _resolve_job(req)
    tdate = req.trade_date or date.today().isoformat()
    asset = req.asset_type or _detect_asset(sym)
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "running", "symbol": sym,
                         "trade_date": tdate, "params": params}
    threading.Thread(target=_run, args=(job_id, sym, tdate, asset, params),
                     daemon=True).start()
    return {"job_id": job_id, "status": "running"}

@app.post("/cancel/{job_id}")
def cancel(job_id: str) -> dict[str, Any]:
    """Request cancellation; the worker stops after its current node."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] == "running":
            job["cancel_requested"] = True
        return {"job_id": job_id, "status": job["status"]}

@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        out: dict[str, Any] = {"job_id": job_id, "status": job["status"]}
        if job.get("error"):
            out["error"] = job["error"]
        if job.get("progress"):
            out["progress"] = job["progress"]
        if job.get("usage"):
            out["usage"] = job["usage"]
        return out

@app.get("/jobs")
def jobs() -> dict[str, Any]:
    """Operator view: in-memory job_id -> status/symbol (resets on restart)."""
    with _lock:
        return {"jobs": {jid: {"status": j.get("status"), "symbol": j.get("symbol", "")}
                         for jid, j in _jobs.items()}}

@app.get("/result/{job_id}")
def result(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] != "done":
            return {"job_id": job_id, "status": job["status"]}
        return {"job_id": job_id, "status": "done", "signal": job["signal"],
                "symbol": job.get("symbol", ""), "trade_date": job.get("trade_date", ""),
                "sections": job["sections"], "params": job.get("params", {}),
                "note": "research-only, no orders"}

@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    cfg = DEFAULT_CONFIG
    return {"provider": cfg.get("llm_provider"), "deep_think_llm": cfg.get("deep_think_llm"),
            "quick_think_llm": cfg.get("quick_think_llm"),
            "analysts": list(ANALYSTS),
            "depths": {"shallow": {"debate_rounds": 1, "risk_rounds": 1},
                       "medium": {"debate_rounds": 2, "risk_rounds": 2},
                       "deep": {"debate_rounds": 3, "risk_rounds": 3}},
            "tiers": ["free", "pro"], "free_model": FREE_MODEL,
            "pro_default_deep": PRO_DEFAULT_DEEP,
            "note": "research-only, no orders"}