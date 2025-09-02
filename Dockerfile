FROM public.ecr.aws/lambda/python:3.11


COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download MiniLM model so Lambda doesn’t fetch it on cold start
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app.py ./

CMD ["app.lambda_handler"]