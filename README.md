# Deploying Hugging Face Models on Amazon SageMaker

This repository contains step-by-step instructions and Python code to deploy open-source Hugging Face models and Large Language Models (LLMs) to Amazon SageMaker endpoints for real-time inference.

---

## Table of Contents
- [Architecture & Overview](#architecture--overview)
- [Prerequisites](#prerequisites)
- [SageMaker Studio & Environment Setup](#sagemaker-studio--environment-setup)
- [Step 1: Session & IAM Role Configuration](#step-1-session--iam-role-configuration)
- [Step 2: Deploying a Task-Specific Model (DistilBERT Example)](#step-2-deploying-a-task-specific-model-distilbert-example)
- [Step 3: Deploying Open-Source LLMs with Hugging Face DLC](#step-3-deploying-open-source-llms-with-hugging-face-dlc)
- [Step 4: Real-Time Inference & Testing](#step-4-real-time-inference--testing)
- [Important: Cost Optimization & Resource Cleanup](#important-cost-optimization--resource-cleanup)

---

## Architecture & Overview
Amazon SageMaker manages the machine learning lifecycle, providing fully managed endpoints for inference. Models can be deployed using:
1. **Hugging Face Hub Integration**: Direct specification of Hugging Face model IDs and task definitions using `HuggingFaceModel`.
2. **Hugging Face Deep Learning Containers (DLC)**: Pre-built Docker container images optimized for large language models (e.g., Falcon models) with GPU acceleration.

---

## Prerequisites
- An active AWS Account.
- Permissions to create IAM roles, SageMaker domains, and S3 buckets.
- Understanding of SageMaker compute pricing (CPU vs. GPU instance tiers).

---

## SageMaker Studio & Environment Setup

1. **Create a SageMaker Domain**:
   - In the AWS Console, navigate to **Amazon SageMaker** > **Domains**.
   - Choose **Set up for single user** (Quick setup).
   - This automatically attaches an execution role with the `AmazonSageMakerFullAccess` policy and configures SageMaker Studio, Canvas, and default storage.
2. **Launch JupyterLab**:
   - Under your user profile in SageMaker Studio, open **JupyterLab**.
   - Create and run a new JupyterLab space (e.g., `test-demo-sagemaker`).
   - Select an appropriate instance type (e.g., `ml.m5.2xlarge` or `ml.m5.xlarge` for development).
3. **Install Dependencies**:
   ```bash
   pip install --upgrade sagemaker boto3
   ```

---

## Step 1: Session & IAM Role Configuration

Initialize the SageMaker session, verify default S3 bucket creation, and retrieve the execution role ARN.

```python
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize SageMaker session and default S3 bucket
sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()

# Retrieve IAM Execution Role
try:
    role = get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SageMaker_Execution_Role')['Role']['Arn']

# Re-instantiate session with configured bucket
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
region = sess.boto_region_name

print(f"SageMaker Role ARN: {role}")
print(f"Session Region: {region}")
```

---

## Step 2: Deploying a Task-Specific Model (DistilBERT Example)

Deploy a question-answering model (`distilbert-base-uncased-distilbert-squad`) from Hugging Face Hub using the SageMaker Python SDK:

```python
from sagemaker.huggingface import HuggingFaceModel

# Hub configuration specifying model ID and task type
hub = {
    'HF_MODEL_ID': 'distilbert-base-uncased-distilbert-squad',
    'HF_TASK': 'question-answering'
}

# Create HuggingFaceModel instance
huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39"
)

# Deploy model to an inference endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

---

## Step 3: Deploying Open-Source LLMs with Hugging Face DLC

For large language models (such as Falcon-40B), retrieve the Hugging Face Deep Learning Container URI and configure multi-GPU instances (e.g., `ml.g5.2xlarge` with NVIDIA A10G GPUs):

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

# Retrieve container image URI
image_uri = get_huggingface_llm_image_uri(
    "huggingface",
    version="0.8.2"
)

# Hub environment configuration for LLM
hub = {
    'HF_MODEL_ID': 'tiiuae/falcon-40b-instruct',
    'SM_NUM_GPUS': '1'  # Specify number of GPUs required
}

# Define model with DLC container
llm_model = HuggingFaceModel(
    image_uri=image_uri,
    env=hub,
    role=role
)

# Deploy to GPU-accelerated instance
llm_predictor = llm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge"
)
```

---

## Step 4: Real-Time Inference & Testing

### Programmatic Invocation via Python SDK
```python
# Payload formatted according to the model task (e.g., Question Answering)
payload = {
    "question": "What is used for inferences?",
    "context": "My name is Krish and I use Amazon SageMaker for model inferences and deployments."
}

response = predictor.predict(payload)
print("Inference Result:", response)
```

### Testing via SageMaker Studio Console
1. Navigate to **Deployments** > **Endpoints** in SageMaker Studio.
2. Select your deployed endpoint.
3. In the **Test endpoint** panel, supply the JSON request body:
   ```json
   {
     "inputs": {
       "question": "What does Krish teach?",
       "context": "My name is Krish and I teach data science and generative AI."
     }
   }
   ```
4. Click **Send request** to inspect real-time responses.

---

## Important: Cost Optimization & Resource Cleanup

AWS SageMaker endpoints incur hourly charges while active. To avoid unexpected bills:
1. **Delete Endpoint via Code**:
   ```python
   predictor.delete_endpoint()
   ```
2. **Delete Endpoint via Console**:
   - Go to **Deployments** > **Endpoints**.
   - Select active endpoints and choose **Actions** > **Delete**.
3. **Stop Compute Spaces**:
   - Stop or delete inactive JupyterLab spaces and associated instances when not in use.
