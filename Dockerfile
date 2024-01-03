# Use the official Python image from the Docker Hub
FROM python:3.8-slim-buster

# Make a directory for our application
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
        procps \
        && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local files into the container
COPY knee_server /app
COPY knee/static /app/knee/static
COPY knee/templates /app/knee/templates
COPY knee/images /app/knee/images
COPY knee_server/columns /app/columns


# Make port 9000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "KNEE_server:app", "--host", "0.0.0.0", "--port", "8000"]