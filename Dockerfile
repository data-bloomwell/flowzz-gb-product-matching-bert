# Stage 1: The Build Stage
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Set the working directory.
WORKDIR /app

# Copy the requirements file.
COPY requirements.txt .

# Install all Python dependencies. Use the --only-binary flag to force pip to
# install pre-compiled wheels for all packages that have them.
RUN pip3 install --no-cache-dir --only-binary :all: -r requirements.txt

# Download the pre-trained model during the build.
RUN python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('models')"

# ---
# Stage 2: The Final Runtime Stage
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory.
WORKDIR /var/task

# Copy the installed Python packages from the 'builder' stage.
COPY --from=builder /var/lang/lib/python3.11/site-packages/ /var/lang/lib/python3.11/site-packages/
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy the downloaded model from the 'builder' stage.
COPY --from=builder /app/models /var/task/models

# Copy your application code.
COPY app.py .

# Set the Lambda handler.
CMD ["app.lambda_handler"]