# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port dynamically based on the PORT environment variable
EXPOSE ${PORT:-80}

# Start the application using Gunicorn with command-line arguments
CMD ["gunicorn", \
     "--bind", "0.0.0.0:${PORT:-80}", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--log-file", "-", \
     "main:app"]