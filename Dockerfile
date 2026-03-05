FROM python:3.11-bookworm

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY dataset /opt/dataset
COPY src /opt/src
COPY start_api_server.sh /opt

WORKDIR /opt
