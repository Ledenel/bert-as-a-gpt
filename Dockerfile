FROM python:3.8

# WORKDIR /app

# RUN apt update

# RUN apt install -y gcc

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/requirements-basic.txt /app/requirements-basic.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements-basic.txt

COPY ./src/init.py /app/init.py

RUN python init.py

WORKDIR /app

COPY ./src/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements.txt

COPY src /app

# RUN python -c "import app"

EXPOSE 5000

ENTRYPOINT [ "flask" ]

CMD [ "run", "--host", "0.0.0.0" ]
