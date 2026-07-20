# Build and run docker container as:
# docker build -t gee_env . && docker run --rm -it -v %CD%:/workdir  gee_env
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir earthengine-api

WORKDIR /workdir
CMD ["/bin/bash"]
