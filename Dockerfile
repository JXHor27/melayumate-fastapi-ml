FROM python:3.11

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install pip first
RUN pip install --upgrade pip

# Install PyTorch + Torchaudio from CPU wheels (much smaller)
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow from prebuilt wheel (avoid huge build temp files)
RUN pip install --no-cache-dir \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.15.0-cp311-cp311-manylinux.whl

# Now install all other requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]

