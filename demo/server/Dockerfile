# Image for a Hugging Face Space (Docker SDK). Built by Hugging Face from the folder that
# tools/build_space.py assembles: app.py, requirements.txt, src/clothseg/ and best.pt.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache DEVICE=cpu MAX_SIDE=1024

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# The network shape (SegFormer-B3 config) is fetched once at build time, so the server
# needs no internet access when it runs.
RUN python -c "from transformers import SegformerConfig; SegformerConfig.from_pretrained('nvidia/mit-b3')" \
 && chmod -R a+rwX /app/.cache
ENV HF_HUB_OFFLINE=1

COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
