FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir httpx openai

# Copy source
COPY postprophet.py .
COPY environments/ environments/
COPY run_duel.py .
COPY test_mock.py .

# Eval set is NOT copied — it's fetched at runtime from the repo
# This ensures the Action always uses the current eval set, not a stale one

# Temperature 0 for deterministic eval
ENV POSTPROPHET_TEMPERATURE=0

# Default: run smoke test (Stage 1)
# Can be overridden for Stage 2 batch eval
CMD ["python", "-c", "print('PostProphet eval container ready')"]
