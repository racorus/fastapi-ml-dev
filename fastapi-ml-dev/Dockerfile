# Base image of Python application
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirement file
COPY ./requirements.txt /app/requirements.txt

# Install requirements packages
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Dump working resources to working directory
COPY ./app /app

# RUn FastAPI app
CMD ["fastapi", "run", "main.py", "--port", "80"]