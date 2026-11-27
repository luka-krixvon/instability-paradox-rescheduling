# linux/amd64 manifest of python:3.9.19-slim-bookworm, resolved 2026-07-29.
FROM python@sha256:70fbdeb75b0d071778ea1df4353ac959118ae8031bc5be33a95d1b8ea1fb8f08

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Keep the image's initial working directory outside the runtime /work tmpfs.
# Otherwise Docker pre-creates /work/snapshot as root before applying --user,
# and the non-root entrypoint cannot populate the verified extraction root.
WORKDIR /snapshot
COPY experiments/journal_p0/repro/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r /tmp/requirements.lock
COPY experiments/journal_p0/repro/container_entrypoint.py /opt/dchhr/container_entrypoint.py

LABEL org.opencontainers.image.title="LLM-DCHHR formal rerun environment" \
      org.opencontainers.image.base.name="python:3.9.19-slim-bookworm" \
      org.opencontainers.image.base.digest="sha256:70fbdeb75b0d071778ea1df4353ac959118ae8031bc5be33a95d1b8ea1fb8f08"

ENTRYPOINT ["python3", "/opt/dchhr/container_entrypoint.py"]
