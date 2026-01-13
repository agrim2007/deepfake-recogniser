FROM python:3.11
WORKDIR /code
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
RUN mkdir -p static/uploads static/spectrograms
RUN chmod 777 static/uploads static/spectrograms
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120"]