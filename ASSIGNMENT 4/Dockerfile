FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy your app code and requirements
COPY . .


# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose the Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]