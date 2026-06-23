#!/bin/bash


echo "Cleaning up port 10025..."
fuser -k 10025/tcp 2>/dev/null || true
sleep 1


echo "Starting Postfix..."
service postfix start


echo "Starting CAPE Scheduler..."
cd /app/CAPEv2
/app/CAPEv2/venv/bin/python3 cuckoo.py &


echo "Starting CAPE Web API..."
/app/CAPEv2/venv/bin/python3 web/manage.py runserver 0.0.0.0:8000 --noreload &


echo "Starting MGW Mail Filter Server on port 10025..."
cd /app/MGW
exec /app/MGW/mail-env/bin/python3 mail_filter.py
