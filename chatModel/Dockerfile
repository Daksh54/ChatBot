# Use a slim Python base image
FROM python:3.10-slim

# Install system dependencies (Tesseract, poppler for PDF, and required libs)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /newLLMtotest

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy rest of the app files
COPY . .

# Expose the Streamlit default port (optional)
EXPOSE 8501

# Set environment variable for logs
ENV PYTHONUNBUFFERED=1

# Run the Streamlit app
CMD ["streamlit", "run", "newLLMtotest.py", "--server.port=8501" , "--server.enableCORS=false", "--server.address=0.0.0.0"]
