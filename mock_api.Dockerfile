# Use a Python 3.10 slim base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Copy the mock API script
COPY mock_api.py .

# Expose the port
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "mock_api:app", "--host", "0.0.0.0", "--port", "8000"]