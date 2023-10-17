FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app file and templates folder
COPY app.py /app/
COPY templates /app/templates

COPY model1.bin /app/model1.bin
COPY dv.bin /app/dv.bin

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
