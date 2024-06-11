# FastAPI App template

___

## Prequisitions

- Docker and Docker Compose

## Pre-Installation

1. Update required python packages in <code>requirements.txt</code>
2. Update expose port of FastAPI app in <code>docker-compose.yml</code> under <code>ports</code> to be <code>{expose_port}:80</code>
3. Copy model file to <code>./model/</code> directory
4. Copy samples training data to <code>./data/</code> directory

## Installation

1. <code>cd</code> to project directory
2. run command <code>docker compose up -d --build</code>
3. check Web UI at <code>{IP:port}</code> as defined
