import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# AWS Bedrock config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BEDROCK_MODEL_ID = os.getenv(
    "AWS_BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-0-20250514"
)

# Which vision model to use: "bedrock" (default), "openai", or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")

# ADB connection
DEVICE_IP = os.getenv("DEVICE_IP", "127.0.0.1")
