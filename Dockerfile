FROM python:3.10-slim

WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt

# Install git and Chromium for SVG export (Chromium + chromedriver versions are matched)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        chromium \
        chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Create chrome symlink for Selenium compatibility and ensure they're in PATH
RUN ln -s /usr/bin/chromium /usr/bin/google-chrome \
    && ln -s /usr/bin/chromedriver /usr/local/bin/chromedriver

# Set Bokeh environment variables for Docker SVG export
ENV BOKEH_CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV BOKEH_IN_DOCKER=1

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Some additional dependencies that seems to be needed for Hugging Face Spaces
RUN pip install --no-cache-dir --upgrade \
    pyconify

COPY . .

# Set up the entrypoint for Hugging Face Spaces
# The port 7860 is the default port that Hugging Face Spaces expects
EXPOSE 7860

CMD ["panel", "serve", "code/patchseq_panel_viz.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin=*"]
