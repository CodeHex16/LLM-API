FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy application files into the container
COPY ./api/llm /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install git
RUN apt-get update && apt-get install -y git

# Expose port (if needed)
EXPOSE 5001

CMD ["pytest", "tests/user_test.py"]
# Start the application
# CMD ["python", "app.py"]
