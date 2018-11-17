FROM tensorflow/tensorflow:1.12.0-py3
COPY ./requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
