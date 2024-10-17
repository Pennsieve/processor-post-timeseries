FROM python:3.12

WORKDIR /processor

RUN apt clean && apt-get update && apt-get -y install libhdf5-dev

COPY processor/requirements.txt /processor/requirements.txt

RUN pip install -r /processor/requirements.txt

COPY processor/ /processor

ENV PYTHONPATH="/"

CMD ["python3.12", "-m", "processor.main"]
