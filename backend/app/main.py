from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import (
    anomaly,
    ask,
    clean,
    clustering,
    dashboard,
    dataset,
    eda,
    export,
    health,
    predict,
    profile,
    report,
    train,
    upload,
)
from app.config import settings
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.middleware.rate_limit import limiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title=settings.app_name, version="1.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(dataset.router)
app.include_router(export.router)
app.include_router(predict.router)
app.include_router(profile.router)
app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(clustering.router)
app.include_router(anomaly.router)
app.include_router(train.router)
app.include_router(ask.router)
app.include_router(dashboard.router)
app.include_router(report.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": "1.2.0"}
