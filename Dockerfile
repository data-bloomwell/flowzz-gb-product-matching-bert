# Use AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

RUN pip install --no-cache-dir "numpy<2"

RUN pip install --no-cache-dir torch==2.2.2 transformers==4.41.2 sentence-transformers==2.7.0

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/var/task/model')"

# Copy dependency file
COPY requirements.txt ./

# Install dependencies (increase timeout/retries to avoid network issues)
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

# Copy your project files
COPY . .

# Set the Lambda handler (module.function)
CMD ["app.lambda_handler"]