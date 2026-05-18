# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the entire application code into the container
COPY --chown=user . /code

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# Command to run the FastAPI app via Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
