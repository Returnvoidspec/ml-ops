FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY classes.json .
COPY model/votre_modele.h5 model/

CMD ["python", "app.py"]
