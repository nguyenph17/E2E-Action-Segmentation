FROM python:3.8-slim
WORKDIR /action

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /action

EXPOSE 5000
CMD ["python", "app.py"]