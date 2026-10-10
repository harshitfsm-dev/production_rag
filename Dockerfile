FROM python:3.12-slim

# uv installs exactly what uv.lock pins, so the image matches a local `uv sync`.
RUN pip install --no-cache-dir uv==0.9.29

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/code/.venv/bin:$PATH" \
    PYTHONPATH=/code

# Dependencies first, in their own layer: editing app code below does not
# reinstall them. --frozen makes a stale lockfile a build error rather than a
# silent re-resolve. --no-install-project because pyproject.toml has no
# [build-system], so uv installs dependencies only and the code is imported from
# /code -- which is what PYTHONPATH above is for. --no-cache keeps uv's download
# cache (as large as the venv itself) out of the image layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

COPY src ./src

# Serving HTTP needs no root. Added after the COPYs so the layers above stay
# root-owned and read-only to the process using them.
RUN useradd --system --uid 1001 app
USER app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
