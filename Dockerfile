FROM python:3.12.0
## pay attention to path !!!!!!!!!!!!!!!!!!!!!
WORKDIR /usr/src/app                             

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "fastapi", "dev", "src/app/main.py", "--host", "0.0.0.0", "--port", "8000" ] 

