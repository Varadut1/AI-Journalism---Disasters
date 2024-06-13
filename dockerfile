FROM python:3.9-slim-buster

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

#
ENV PYTHONUNBUFFERED=1

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "webapp.py"]
