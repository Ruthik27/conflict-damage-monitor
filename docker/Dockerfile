FROM python:3.10-slim

# GDAL system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source (layer cache)
COPY environment.yml /tmp/environment.yml
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic python-multipart \
    rasterio shapely geopandas \
    segmentation-models-pytorch timm \
    pytorch-lightning torchmetrics \
    pyyaml python-dotenv sqlalchemy geoalchemy2 psycopg2-binary

COPY src/ /app/src/
COPY configs/ /app/configs/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
