FROM pytorch/pytorch

WORKDIR /app

RUN apt update

RUN apt install -y gcc

COPY ./cache /app/cache

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/requirements.txt /app/requirements.txt

RUN pip install -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements.txt

COPY src /app

# RUN python -c "import app"

EXPOSE 5000

ENTRYPOINT [ "flask" ]

CMD [ "run", "--host", "0.0.0.0" ]
