"""FastAPI entry point.

uvicorn infrastructure.api.main:app --reload
python -m infrastructure.api.main
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from infrastructure.api.dependencies import Container, build_container, get_container
from infrastructure.api.settings import Settings
from infrastructure.api.v1.endpoints.authors import router as authors_router
from infrastructure.api.v1.endpoints.recommend import router as recommend_router
from infrastructure.schemas.recommendation import HealthResponse


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or build_container(settings)
        yield

    app = FastAPI(
        title="Ethical Recommender Service",
        version="0.1.0",
        lifespan=lifespan,
        description="Recomendación ética de investigadores para Centinela V3",
    )
    if settings.cors_origins:
        # Browser clients on another origin (centinela-front on :4200). With the ng-serve proxy
        # (/api/recommender -> :8000) requests are same-origin and CORS is not involved.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            max_age=600,
        )
    app.include_router(recommend_router)
    app.include_router(authors_router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health(c: Container = Depends(get_container)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=c.embeddings.model_name,
            authors=len(c.embeddings.author_ids),
            embedding_dim=int(c.embeddings.matrix.shape[1]),
            metadata_source=c.metadata.source,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "infrastructure.api.main:app",
        host=os.environ.get("RECO_HOST", "127.0.0.1"),
        port=int(os.environ.get("RECO_PORT", "8000")),
    )
