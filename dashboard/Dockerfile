FROM python:3.11-slim

WORKDIR /app

# Copy entire project (need src/ for imports)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r dashboard/requirements.txt

# HuggingFace Spaces expects port 7860
ENV PORT=7860
ENV DASH_DEBUG=0
ENV DASH_CACHE_DIR=/tmp/dash_cache
EXPOSE 7860

WORKDIR /app/dashboard
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
