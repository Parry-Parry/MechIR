
FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.in .
RUN pip install pip-tools && pip-compile requirements.in && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace/src

CMD ["pytest"]