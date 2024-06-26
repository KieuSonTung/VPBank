FROM python:3.9.6-slim

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y gcc default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install libgomp1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /workspace/src/
COPY weights/ /workspace/weights/

WORKDIR /workspace

