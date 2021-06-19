FROM python:2

COPY req.txt .

RUN pip install --no-cache-dir -r req.txt
