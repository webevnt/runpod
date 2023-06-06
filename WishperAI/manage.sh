#!/bin/bash

# set -o errexit
# set -o nounset

# alembic upgrade head
# uvicorn app.main:app --port=8081 --host 0.0.0.0
gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:8081
