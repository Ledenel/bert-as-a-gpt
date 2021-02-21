FROM ledenel/zh-nlp-basic:main

WORKDIR /app

COPY ./src/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements.txt

COPY src /app

# RUN python -c "import app"

EXPOSE 5000

ENTRYPOINT [ "flask" ]

CMD [ "run", "--host", "0.0.0.0" ]
