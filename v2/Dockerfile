FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_ROOT=/app/dataset
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
