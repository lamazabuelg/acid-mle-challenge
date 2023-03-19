FROM python:3.9.16-slim-buster
ENV PYTHONUNBUFFERED True
COPY . ./
COPY ./requirements.txt /src/requirements.txt
WORKDIR /src
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
