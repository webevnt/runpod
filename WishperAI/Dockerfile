FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /
COPY requirements.txt ./
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade pip
RUN pip3 install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --no-cache-dir -r requirements.txt
RUN rm requirements.txt

COPY . /app
WORKDIR /app

RUN chmod +x ./manage.sh

EXPOSE 8081
