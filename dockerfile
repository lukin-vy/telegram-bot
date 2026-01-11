FROM python:3.12

WORKDIR /app

COPY ./app/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel
RUN pip install -r requirements.txt

COPY ./app ./

CMD ["python", "main.py"]