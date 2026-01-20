FROM python:3.11-slim

WORKDIR /app

COPY app /app

EXPOSE 3333

CMD ["python", "server.py"]
