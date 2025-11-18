FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict.py .

EXPOSE 9696

#CMD ["python", "predict.py"]
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]