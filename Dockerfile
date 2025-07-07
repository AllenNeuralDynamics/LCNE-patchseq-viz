FROM python:3.10-slim

WORKDIR /code

# Set environment variables (solve numba caching issue of UMAP)
ENV NUMBA_DISABLE_CACHE=1

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Some additional dependencies that seems to be needed for Hugging Face Spaces
RUN pip install --no-cache-dir --upgrade \
    pyconify

COPY . .

# Install the package
RUN pip install --no-cache-dir .

# Set up the entrypoint for Hugging Face Spaces
# The port 7860 is the default port that Hugging Face Spaces expects
CMD panel serve src/LCNE_patchseq_analysis/panel_app/patchseq_panel_viz.py \
    --address 0.0.0.0 \
    --port 7860 \
    --allow-websocket-origin="*" \
