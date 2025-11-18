RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use an official Python runtime as a parent image
FROM python:3.11-slim
RUN pip install --upgrade pip

RUN adduser -u 1234 myuser
USER myuser
# # Install system dependencies (for soundfile & malaya)
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (for layer caching)
COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of your application code into the container
COPY . .

# Tell Docker what port the app will run on
EXPOSE 8000

# Command to run the app using Hypercorn. This replaces your Procfile.
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]