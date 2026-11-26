# TokenForge gateway image, plus a "fake" target with the fixed-timing fake engine.
#
# Build from the repo root (the build context is gateway/, so venv/ and results stay out):
#     docker build --target gateway -t tokenforge-gateway gateway/
#     docker build --target fake -t tokenforge-fake-engine gateway/

# Stage 1: install dependencies into a virtualenv. Only /opt/venv leaves this stage,
# so pip and its caches never reach the final image.
FROM python:3.14.10-slim-trixie AS build
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt

# Stage 2: the runtime image.
FROM python:3.14.10-slim-trixie AS gateway
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY __init__.py app.py keys.py gateway/
# A fixed numeric UID, so Kubernetes can check runAsNonRoot without reading /etc/passwd.
RUN useradd --system --no-create-home --uid 10001 gateway
USER 10001
EXPOSE 8080
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]

# Test-only image: the gateway image plus the fake engine, so the stack runs without a GPU.
FROM gateway AS fake
COPY fake_upstream.py gateway/
EXPOSE 8000
CMD ["uvicorn", "gateway.fake_upstream:app", "--host", "0.0.0.0", "--port", "8000"]
