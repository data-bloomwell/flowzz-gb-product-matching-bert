# Stage 1: The Build Stage
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install system dependencies
RUN yum update -y && yum install -y \
    gcc libffi-devel openssl-devel make cmake \
    libxml2-devel libxslt-devel libcurl-devel \
    postgresql-devel mysql-devel \
    && yum clean all

# Set the working directory
WORKDIR /app

# Copy the requirements file and install packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-package the MiniLM model directly into the builder container
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save_pretrained('/app/minilm_model')"

# ---
# Stage 2: The Final, Lean Stage
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory
WORKDIR /var/task

# Copy only the installed Python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /var/lang/lib/python3.11/site-packages/

# Copy the pre-packaged MiniLM model
COPY --from=builder /app/minilm_model /var/task/minilm_model

# Copy your application code
COPY . .

# Set the entry point to your Lambda function's handler
CMD ["app.lambda_handler"]