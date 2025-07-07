FROM python:3.10-slim

WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt

# Install git
RUN apt-get update && apt-get install -y git

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Some additional dependencies that seems to be needed for Hugging Face Spaces
RUN pip install --no-cache-dir --upgrade \
    pyconify

COPY . .

# Set up the entrypoint for Hugging Face Spaces
# The port 7860 is the default port that Hugging Face Spaces expects
CMD panel serve code/patchseq_panel_viz.py \
    --address 0.0.0.0 \
    --port 7860 \
    --allow-websocket-origin="*" \
