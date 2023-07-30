# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory (with your app files) into the container
COPY . .

# Expose the port that FastAPI will run on (change the port number if needed)
EXPOSE 8000

# Run the Uvicorn server to serve the FastAPI app
CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8000"]


