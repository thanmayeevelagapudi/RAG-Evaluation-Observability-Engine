"""FastAPI application entry point with startup/shutdown lifecycle."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import routes
from api.dependencies import close_dependencies, get_updater, init_dependencies
from config.settings import get_settings

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )


def _upgrade_head() -> None:
    from alembic import command as alembic_command
    from alembic.config import Config
    cfg = Config(str(Path(__file__).parent.parent / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    alembic_command.upgrade(cfg, "head")


async def _run_migrations() -> None:
    """Run Alembic migrations at startup in a worker thread.

    alembic/env.py calls asyncio.run() internally, which raises RuntimeError
    if executed on the loop that's already running the FastAPI lifespan —
    running it in a plain thread via run_in_executor avoids that conflict.
    """
    try:
        await asyncio.get_running_loop().run_in_executor(None, _upgrade_head)
        log.info("Database migrations up to date")
    except Exception as exc:
        log.warning("Alembic migration skipped (%s) — ensure DB schema is current", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    await _run_migrations()
    await init_dependencies()

    updater = get_updater()
    update_task = asyncio.create_task(
        updater.start_background_loop(interval_seconds=60)
    )
    log.info("Application started — all systems ready")

    yield

    update_task.cancel()
    try:
        await update_task
    except asyncio.CancelledError:
        pass

    await close_dependencies()
    log.info("Application shutdown complete")


app = FastAPI(
    title="Adaptive RAG Retrieval & Eval Platform",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Auth is a static X-Api-Key header, not cookies — credentials aren't needed,
    # and combining allow_credentials with a "*" origin is a CORS anti-pattern.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-Api-Key"],
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics(
    x_api_key: str | None = Header(default=None),
    x_metrics_key: str | None = Header(default=None),
) -> Response:
    # Optional protection: disabled by default for local/dev, enabled for public deployments.
    if settings.metrics_require_api_key:
        expected = settings.metrics_api_key or settings.api_key
        provided = x_metrics_key or x_api_key
        if not expected or provided != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing metrics API key")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
