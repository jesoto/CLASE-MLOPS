FROM python:3.11-slim

WORKDIR /app_fastapi

#COPY . .

COPY requirements.txt .
COPY app_fastapi.py ./
COPY models ./models

EXPOSE 8000


RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
