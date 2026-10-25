FROM python:3.11-slim

WORKDIR /app
ARG EMBEDDING_PROFILE=semcse
ENV EMBEDDING_PROFILE=${EMBEDDING_PROFILE}

# Dependencies are installed at build time; nothing is installed or downloaded at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Isolate model provisioning from application changes so rebuilding an agent does not redownload weights.
COPY litagent/config.py litagent/trace.py litagent/__init__.py ./litagent/
COPY litagent/integrations/embeddings.py litagent/integrations/__init__.py ./litagent/integrations/
COPY scripts/setup_embeddings.py ./scripts/setup_embeddings.py
RUN python scripts/setup_embeddings.py
COPY . .
RUN mkdir -p /app/logs

# Zetaris and Nightshift run outside this container: point their URLs at
# host.docker.internal instead of localhost (see .env.example).
ENV PORT=8000
EXPOSE 8000

CMD ["python", "run.py"]
