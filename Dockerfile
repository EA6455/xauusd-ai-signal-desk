FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "gunicorn -w 2 --threads 16 -b 0.0.0.0:${PORT:-7860} --timeout 120 app:app"]
