# Use AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

RUN pip install --no-cache-dir --upgrade pip

ENV HF_HOME=/tmp/huggingface

# Copy dependency file
COPY requirements.txt ./

# Install dependencies (increase timeout/retries to avoid network issues)
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/var/task/model')"

# Copy your project files
COPY . .

# Set the Lambda handler (module.function)
CMD ["app.lambda_handler"]