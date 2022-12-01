FROM python:3.9-slim
WORKDIR /action

RUN apt-get update

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /action

EXPOSE 5000
CMD ["python", "app.py"]