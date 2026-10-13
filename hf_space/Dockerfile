FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user app.py .
EXPOSE 7860
CMD ["python", "app.py"]
