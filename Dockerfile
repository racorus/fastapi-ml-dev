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

# Copy app and models
COPY ./app /app
COPY ./model /app/model
COPY ./samples_training_data /app/samples_training_data

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
