"""Backend for the demo web UI.

Runs the same review graph the webhook uses, but against a diff supplied
directly by the browser instead of one fetched from GitHub. That means the demo
page works with no repo, no webhook, and no GitHub token.

Progress is streamed back as Server-Sent Events so the page can light up each
agent as it finishes rather than waiting on one long request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import compile_graph
from app.agents.state import ALL_SPECIALISTS, FileChange, new_state

logger = logging.getLogger("app.webui")

router = APIRouter()

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures"
_GOLDEN_SET = _FIXTURE_DIR.parent / "golden_set.json"

# Diffs bigger than this are rejected outright — the demo is not a bulk tool.
_MAX_DIFF_CHARS = 20_000


class ReviewRequest(BaseModel):
    """A diff pasted into the demo page."""

    diff: str = Field(min_length=1, max_length=_MAX_DIFF_CHARS)
    title: str = "Untitled change"


@lru_cache
def _graph():
    """Compile once, reuse across requests."""
    return compile_graph()


def _filename_from_diff(diff: str) -> str:
    """Pull the target path out of a unified diff's `+++ b/...` header."""
    match = re.search(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
    return match.group(1).strip() if match else "changed_file.py"


def _line_counts(diff: str) -> tuple[int, int]:
    """Count added/removed lines, ignoring the file headers."""
    added = sum(
        1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    )
    removed = sum(
        1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")
    )
    return added, removed


def _jsonable(value: Any) -> Any:
    """Convert Pydantic models (and containers of them) into plain JSON types."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sse(event: str, payload: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.get("/api/fixtures")
async def list_fixtures() -> list[dict]:
    """Expose the eval fixtures as one-click examples for the demo page."""
    try:
        golden = json.loads(_GOLDEN_SET.read_text())
    except (OSError, json.JSONDecodeError):
        golden = {}

    items: list[dict] = []
    for path in sorted(_FIXTURE_DIR.glob("*.diff")):
        meta = golden.get(path.name, {})
        items.append(
            {
                "name": path.name,
                "label": path.stem.replace("_", " "),
                "kind": meta.get("kind", "unknown"),
                "intent": meta.get("intent", ""),
                "diff": path.read_text(),
            }
        )
    # Show the vulnerable ones first — they make the better demo.
    items.sort(key=lambda i: (i["kind"] != "insecure", i["label"]))
    return items


async def _run_events(req: ReviewRequest) -> AsyncIterator[str]:
    """Drive the graph, emitting an SSE frame as each node reports back."""
    filename = _filename_from_diff(req.diff)
    added, removed = _line_counts(req.diff)

    state = new_state(
        owner="demo",
        repo="demo",
        pr_number=0,
        pr_title=req.title,
        pr_body="Submitted from the demo UI.",
        diff=req.diff,
        changed_files=[
            FileChange(
                filename=filename,
                status="modified",
                additions=added,
                deletions=removed,
                changes=added + removed,
                patch=req.diff,
            )
        ],
    )

    yield _sse("start", {"file": filename, "additions": added, "deletions": removed})

    started = time.perf_counter()
    merged: dict[str, Any] = {}

    try:
        async for chunk in _graph().astream(state, stream_mode="updates"):
            for node, update in (chunk or {}).items():
                update = update or {}
                # `errors` and `token_usage` accumulate; everything else replaces.
                for key, value in update.items():
                    if key == "errors":
                        merged.setdefault("errors", []).extend(value or [])
                    elif key == "token_usage":
                        merged.setdefault("token_usage", {}).update(value or {})
                    else:
                        merged[key] = value

                payload = {"node": node, "update": _jsonable(update)}
                if node == "coordinator":
                    payload["agents_to_run"] = update.get(
                        "agents_to_run", list(ALL_SPECIALISTS)
                    )
                yield _sse("node", payload)
    except Exception as exc:  # noqa: BLE001 - surface the failure to the browser
        logger.exception("demo review failed")
        yield _sse("error", {"message": str(exc)})
        return

    usage = merged.get("token_usage", {}) or {}
    yield _sse(
        "done",
        {
            "final_review": merged.get("final_review"),
            "security_findings": _jsonable(merged.get("security_findings")),
            "diff_findings": _jsonable(merged.get("diff_findings")),
            "test_suggestions": _jsonable(merged.get("test_suggestions")),
            "errors": merged.get("errors", []),
            "stats": {
                "agents": len(usage),
                "tokens": sum(u.get("total", 0) for u in usage.values()),
                "cost_usd": round(
                    sum(u.get("cost_usd", 0.0) for u in usage.values()), 6
                ),
                "latency_s": round(time.perf_counter() - started, 1),
            },
        },
    )


@router.post("/api/review/stream")
async def review_stream(req: ReviewRequest) -> StreamingResponse:
    """Review a pasted diff, streaming per-agent progress as SSE."""
    return StreamingResponse(
        _run_events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
