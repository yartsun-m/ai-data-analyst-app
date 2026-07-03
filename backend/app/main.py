from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ask, clean, dashboard, dataset, eda, health, profile, report, train, upload
from app.config import settings

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(dataset.router)
app.include_router(profile.router)
app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(train.router)
app.include_router(ask.router)
app.include_router(dashboard.router)
app.include_router(report.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
