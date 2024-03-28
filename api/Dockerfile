# Use the official Python image as the base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and the trained model into the container
COPY app.py trained_model.pkl /app/

# RUN pip install additional_dependency
RUN pip install --upgrade pip

# Expose port 5000
EXPOSE 5000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]