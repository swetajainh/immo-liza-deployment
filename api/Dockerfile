# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app


# install the dependencies
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install  -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app



# Run app.py when the container launches
CMD ["uvicorn","app:app","--host","0.0.0.0"]
