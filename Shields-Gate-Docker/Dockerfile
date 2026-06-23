FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    postfix \
    psmisc \
    python3 \
    python3-pip \
    python3-venv \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY ./MGW ./MGW
COPY ./CAPEv2 ./CAPEv2


RUN python3 -m venv /app/MGW/mail-env && \
    /app/MGW/mail-env/bin/pip install --upgrade pip && \
    /app/MGW/mail-env/bin/pip install --no-cache-dir -r /app/MGW/requirements.txt


RUN python3 -m venv /app/MGW/mail-env && \
    /app/MGW/mail-env/bin/pip install --upgrade pip && \
    /app/MGW/mail-env/bin/pip install --no-cache-dir -r /app/MGW/requirements.txt


COPY postfix_config/main.cf /etc/postfix/main.cf
COPY postfix_config/master.cf /etc/postfix/master.cf
COPY postfix_config/transport /etc/postfix/transport


RUN postmap /etc/postfix/transport


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


EXPOSE 25 587 10025 10026 8000

CMD ["/entrypoint.sh"]
