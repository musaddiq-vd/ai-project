import boto3
import json

from botocore.config import Config

# Retry configuration for handling Throttling (429 throttling Exception)
retry_config = Config(
    retries = {
        'max_attempts': 5,
        'mode': 'adaptive' 
    }
) 

# Bedrock Client
bedrock = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-1",
    config=retry_config  # for throttling remove comment for expo backoff
)

# Knowledge Base ID
KNOWLEDGE_BASE_ID = "OYFJJVWQST"

# Foundation Model ARN
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"

# Guardrail Details
GUARDRAIL_ID = "p2oe3l1jd89w"
GUARDRAIL_VERSION = "1"


def lambda_handler(event, context):

    try:

        # Handle OPTIONS request
        if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": ""
            }

        # Parse request body
        body = json.loads(event.get("body", "{}"))

        # Get user question
        question = body.get("question", "").strip()

        # Validate input
        if not question:
            return {
                "statusCode": 400,
                "headers": cors_headers(),
                "body": json.dumps({
                    "error": "Question is required"
                })
            }

        # Retrieve and Generate
        response = bedrock.retrieve_and_generate(
            input={
                "text": question
            },
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                
                    # Yahan Hybrid Search define karein
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5, # Top 5 chunks
                            "overrideSearchType": "HYBRID" # Yahan 'HYBRID' ya 'SEMANTIC' likhte hain
                        }
                    },

                    # Guardrails
                    "generationConfiguration": {
                        "guardrailConfiguration": {
                            "guardrailId": GUARDRAIL_ID,
                            "guardrailVersion": GUARDRAIL_VERSION
                        }
                    }
                }
            }
        )

        # Extract answer
        answer = response["output"]["text"]

        # Extract citations / sources
        sources = []

        if "citations" in response:

            for citation in response["citations"]:

                retrieved_refs = citation.get(
                    "retrievedReferences",
                    []
                )

                for ref in retrieved_refs:

                    location = ref.get("location", {})

                    s3_uri = location.get(
                        "s3Location",
                        {}
                    ).get(
                        "uri",
                        "Unknown source"
                    )

                    text_preview = ref.get(
                        "content",
                        {}
                    ).get(
                        "text",
                        ""
                    )

                    sources.append({
                        "source": s3_uri,
                        "preview": text_preview[:200]
                    })

        # Success response
        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": json.dumps({
                "answer": answer,
                "sources": sources
            })
        }

    except Exception as e:

        # Error response
        return {
            "statusCode": 500,
            "headers": cors_headers(),
            "body": json.dumps({
                "error": str(e)
            })
        }


# CORS Headers
def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "OPTIONS,POST"
    }