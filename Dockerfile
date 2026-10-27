FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECO_HOST=0.0.0.0 \
    RECO_PORT=8007

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser application ./application
COPY --chown=appuser:appuser domain ./domain
COPY --chown=appuser:appuser infrastructure ./infrastructure

USER appuser
EXPOSE 8007
CMD ["sh", "-c", "exec uvicorn infrastructure.api.main:app --host \"$RECO_HOST\" --port \"$RECO_PORT\""]
