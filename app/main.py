"""FastAPI application entrypoint.

Phase 1 scope: config validation on startup, structured logging with a
per-request id, and a health check. Webhook routes are added in Phase 6.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.cache import claim_review, close_redis
from app.config import ConfigError, get_settings
from app.github.webhook import parse_pull_request_event, verify_signature
from app.logging_config import request_id_ctx, setup_logging
from app.observability.stats import stats
from app.review import run_review
from app.webui import router as webui_router

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration and configure logging before serving traffic."""
    try:
        settings = get_settings()
    except ConfigError:
        # get_settings already printed a readable message to stderr.
        raise

    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "startup complete",
        extra={"version": __version__, "llm_provider": settings.LLM_PROVIDER},
    )
    yield
    await close_redis()
    logger.info("shutdown complete")


app = FastAPI(
    title="Autonomous PR Reviewer",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a correlation id to every request and echo it in the response."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_ctx.set(rid)
    try:
        logger.info(
            "request received",
            extra={"method": request.method, "path": request.url.path},
        )
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# Demo UI: static assets plus the routes that drive them.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
app.include_router(webui_router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the demo page."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe."""
    return JSONResponse({"status": "ok", "version": __version__})


@app.get("/stats")
async def get_stats() -> JSONResponse:
    """Aggregate review metrics: PRs reviewed, avg tokens/cost/latency per PR."""
    return JSONResponse(stats.snapshot())


@app.post("/webhook/github")
async def github_webhook(
    request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Verify, dedup, and enqueue an automated review for a PR event.

    Returns 200 quickly (within GitHub's timeout) and runs the review in the
    background. Invalid signatures are rejected with 401.
    """
    settings = get_settings()
    body = await request.body()

    if not verify_signature(
        settings.GITHUB_WEBHOOK_SECRET,
        body,
        request.headers.get("X-Hub-Signature-256"),
    ):
        logger.warning("webhook rejected: invalid signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    event = parse_pull_request_event(payload)
    if event is None:
        return JSONResponse({"status": "ignored"})

    # Dedup on head SHA so re-delivered/duplicate events review only once.
    key = f"reviewed:{event.owner}/{event.repo}#{event.pr_number}@{event.head_sha}"
    if not await claim_review(key):
        logger.info("webhook duplicate skipped", extra={"sha": event.head_sha})
        return JSONResponse({"status": "duplicate", "sha": event.head_sha})

    background_tasks.add_task(
        run_review, event.owner, event.repo, event.pr_number
    )
    logger.info(
        "webhook accepted; review queued",
        extra={
            "repo": f"{event.owner}/{event.repo}",
            "pr_number": event.pr_number,
            "action": event.action,
            "sha": event.head_sha,
        },
    )
    return JSONResponse(
        {"status": "accepted", "pr": event.pr_number, "sha": event.head_sha}
    )
