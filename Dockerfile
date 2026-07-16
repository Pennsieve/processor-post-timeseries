FROM python:3.12

WORKDIR /processor

RUN apt clean && apt-get update && apt-get -y install libhdf5-dev

COPY processor/requirements.txt /processor/requirements.txt

RUN pip install -r /processor/requirements.txt

COPY processor/ /processor

# pynwb/platformdirs needs a writable cache dir even when the container
# runs as a non-root user (HOME may be unset/"/" in the compute env)
ENV XDG_CACHE_HOME=/tmp/.cache

ENV PYTHONPATH="/"

CMD ["python3.12", "-m", "processor.main"]
