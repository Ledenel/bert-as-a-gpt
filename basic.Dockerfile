FROM python:3.8

# WORKDIR /app

# RUN apt update

# RUN apt install -y gcc

COPY ./cache /app/cache

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/requirements-basic.txt /app/requirements-basic.txt

WORKDIR /app

RUN pip install --no-cache-dir -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements-basic.txt

