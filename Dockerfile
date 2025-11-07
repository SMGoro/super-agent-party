FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y gcc curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \ 
    apt-get install -y nodejs

WORKDIR /app

COPY pyproject.toml ./
COPY package.json ./

RUN pip install uv && \
    uv venv && \
    uv sync

RUN npm install --production --legacy-peer-deps

RUN cd static && npm install --production --legacy-peer-deps && cd ..

COPY . .

RUN mkdir -p uploaded_files && \
    chmod 755 uploaded_files

EXPOSE 3456
ENV HOST=0.0.0.0 PORT=3456 PYTHONUNBUFFERED=1

CMD [".venv/bin/python", "server.py", "--host", "0.0.0.0", "--port", "3456"]
