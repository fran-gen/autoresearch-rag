FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=300
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8001 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
