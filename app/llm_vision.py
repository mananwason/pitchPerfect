import base64
import json
from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    AWS_REGION,
    AWS_BEDROCK_MODEL_ID,
)

SYSTEM_PROMPT = """You analyze Hinge dating app screenshots and generate personalized opening messages.

For each screenshot, respond with JSON only:
{
  "decision": "like" or "skip",
  "comment": "your message" or null,
  "reasoning": "why"
}

Rules:
- "like" if there are conversation hooks (prompts, interesting photos, shared interests)
- "skip" if the profile is empty/low-effort/no hooks
- Comments must be 1-2 sentences, under 280 chars
- Reference something SPECIFIC from their profile
- Be witty and natural, not generic
- No pickup lines, no emojis, no "Hey beautiful"
- Never mention you're an AI
- JSON only, no markdown"""


def encode_screenshot(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return raw


def analyze_profile_bedrock(image_path: str) -> dict:
    """Send screenshot to Claude on AWS Bedrock using the Converse API with inference profiles."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Determine media type
    if image_path.endswith(".png"):
        media_format = "png"
    elif image_path.endswith(".gif"):
        media_format = "gif"
    else:
        media_format = "jpeg"

    # Use the Converse API — works with inference profiles (us.anthropic.* IDs)
    response = client.converse(
        modelId=AWS_BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": media_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {
                        "text": "Analyze this Hinge profile and generate a response."
                    },
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.8,
        },
    )

    raw = response["output"]["message"]["content"][0]["text"]
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile_openai(image_path: str) -> dict:
    """Send screenshot to GPT-4o Vision and get like/skip + comment."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    b64 = encode_screenshot(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this Hinge profile and generate a response.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        max_tokens=300,
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile_gemini(image_path: str) -> dict:
    """Send screenshot to Gemini 1.5 Flash Vision and get like/skip + comment."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    img = Image.open(image_path)
    response = model.generate_content(
        [
            SYSTEM_PROMPT
            + "\n\nAnalyze this Hinge profile and generate a response.",
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0.8,
            max_output_tokens=300,
        ),
    )

    raw = response.text.strip()
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile(image_path: str) -> dict:
    """Route to configured LLM provider."""
    if LLM_PROVIDER == "bedrock":
        return analyze_profile_bedrock(image_path)
    elif LLM_PROVIDER == "gemini":
        return analyze_profile_gemini(image_path)
    elif LLM_PROVIDER == "openai":
        return analyze_profile_openai(image_path)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use bedrock/openai/gemini.")