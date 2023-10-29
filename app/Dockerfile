# Dockerfile
FROM python:3.10.2
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

CMD ["python", "-u", "api.py"]