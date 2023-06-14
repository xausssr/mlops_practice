# syntax=docker/dockerfile:1
FROM python:3.10-slim-buster

WORKDIR /lab2

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD python3 data_preprocessing.py ; python3 model_fit.py ; python3 .model_estimate.py