# Base image of Python application
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirement file
COPY ./requirements.txt /app/requirements.txt

# Upgrade setuptools and wheel
RUN pip install --no-cache-dir --upgrade setuptools wheel

# Install requirements packages
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Dump working resources to working directory
COPY ./app /app

# Copy models
COPY ./model /app/model

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
