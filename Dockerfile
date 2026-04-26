FROM python:3.11-slim

# HuggingFace Spaces expects user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Fix permissions
RUN chown -R user:user /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

USER user

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
