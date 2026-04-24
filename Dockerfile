FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]