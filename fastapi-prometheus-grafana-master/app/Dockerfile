FROM python:3.11-slim

WORKDIR /app

# Ensure required libraries for OpenCV and PyTorch, and git for pip install
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev git libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./root_requirements.txt
COPY fastapi-prometheus-grafana-master/app/requirements.txt ./app_requirements.txt

# Install dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r root_requirements.txt
RUN pip install -r app_requirements.txt

COPY . .

WORKDIR /app/fastapi-prometheus-grafana-master
CMD uvicorn --host=0.0.0.0 app.main:app
