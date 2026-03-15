FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 5000

CMD ["sh", "-c", "python init_db.py && gunicorn -w 2 -b 0.0.0.0:5000 --timeout 0 app:app"]