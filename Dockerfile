# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1


RUN apt-get update \
  && apt-get -y install netcat gcc postgresql \
  && apt-get clean


RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# Environment log level.  Options: notset, debug, info, warning, error, critical
ENV LOG_LEVEL=info

# CMD ["python", "main.py"]
# FastAPI
CMD ["python", "-m", "src"]

# docker build -t api-demo .
# docker run -d -p 8000:8000 --name optum-1 api-demo
