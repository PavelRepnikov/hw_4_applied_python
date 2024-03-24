FROM python:3

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8000

RUN apt-get update && \
    apt-get install -y --no-install-recommends redis-server && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "service redis-server start && uvicorn main:app --host 0.0.0.0 --port 8000"]
