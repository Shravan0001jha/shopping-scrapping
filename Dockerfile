# Use the official Python image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install  -r requirements.txt
RUN python -m spacy download en_core_web_sm


# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]