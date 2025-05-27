
FROM bitnami/spark:3.4.1


WORKDIR /app



USER root
RUN apt-get update && \
    apt-get install -y zip && \
    rm -rf /var/lib/apt/lists/*



COPY __init__.py .



COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



COPY main.py .
COPY src ./src



ENV PYTHONPATH="/app:${PYTHONPATH}"








