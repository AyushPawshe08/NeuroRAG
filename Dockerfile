FROM python:3.11

WORKDIR /app

# Copy only requirements first, so Docker caches this layer and doesn't
# reinstall every package every time you change your Python code.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (config.py, embedder.py, main.py, etc.)
COPY . .

EXPOSE 8000

# Use uvicorn directly, matching how you already run this locally.
# --host 0.0.0.0 is required in Docker - "localhost" inside a container
# only listens to itself, not to traffic from outside the container.
# ${PORT:-8000} uses the platform's assigned port if set (Render/Railway do
# this), otherwise falls back to 8000 for local testing.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}