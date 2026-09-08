FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
WORKDIR /app
COPY ai /app/ai
RUN pip install --no-cache-dir -c ai/requirements-serve.lock './ai[serve]' && useradd --create-home --uid 10001 ostosense
USER ostosense
EXPOSE 8000
CMD ["python", "-m", "ostosense_ai.service"]
