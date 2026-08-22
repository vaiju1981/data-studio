# Build the wheel separately so the runtime image carries no toolchain.
FROM python:3.10-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Reading a path off the host is right for a local tool and wrong for a shared
    # one, so the container turns it off and an operator must opt back in.
    SDS_ALLOW_LOCAL_PATHS=false \
    SDS_LOG_FORMAT=json \
    SDS_DUCKDB_TEMP_DIR=/workspace/duckdb

COPY --from=build /dist/*.whl /tmp/
COPY requirements.lock /tmp/requirements.lock
# The lockfile first and on its own, so every version is the one CI tested. The
# wheel then installs with its dependencies already satisfied — resolving them
# afresh from pyproject is how a container ends up running a release nothing was
# ever run against.
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
    && pip install --no-cache-dir --no-deps /tmp/*.whl \
    && rm /tmp/*.whl /tmp/requirements.lock \
    && useradd --create-home --uid 10001 studio \
    && mkdir -p /workspace/duckdb && chown -R studio:studio /workspace

USER studio
WORKDIR /workspace

# The root filesystem can be mounted read-only; everything written at runtime —
# DuckDB spill and Streamlit's own state — lives under /workspace.
VOLUME ["/workspace"]
EXPOSE 8501

# Liveness is Streamlit's own endpoint; readiness additionally proves DuckDB works
# and that the configured model is actually served.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -m smart_data_studio.healthcheck || exit 1

# The app path is resolved from the installed package rather than hard-coded, so a
# Python version bump does not silently move it out from under us.
ENTRYPOINT ["sh", "-c", "exec streamlit run \
    \"$(python -c 'import smart_data_studio.ui.app as app; print(app.__file__)')\" \
    --server.port=8501 --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false"]
