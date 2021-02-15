FROM python:3.8.7

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements.txt

COPY . /app

# RUN python -c "import app"

ENTRYPOINT [ "flask" ]

EXPOSE 5000

CMD [ "run", "--host", "0.0.0.0" ]
