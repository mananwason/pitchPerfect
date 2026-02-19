import base64
import json
from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    AWS_REGION,
    AWS_BEDROCK_MODEL_ID,
)

SYSTEM_PROMPT = """You analyze Hinge dating app profile screenshots and generate personalized opening messages.

You will receive MULTIPLE screenshots of the SAME profile, scrolled from top to bottom.
This gives you the full picture: all photos, all prompts, all details.

For each profile, respond with JSON only:
{
  "decision": "like" or "skip",
  "comment": "your message" or null,
  "target_screenshot": 1-based index of which screenshot contains the prompt/photo you're referencing,
  "reasoning": "why"
}

Rules:
- "like" if there are conversation hooks (prompts, interesting photos, shared interests)
- "skip" if the profile is empty/low-effort/no hooks
- Comments must be 1-2 sentences, under 280 chars
- Reference something SPECIFIC you saw in one of the screenshots
- Set target_screenshot to indicate WHICH screenshot has the content you're commenting on
- Be witty and natural, not generic
- No pickup lines, no emojis, no "Hey beautiful"
- Never mention you're an AI
- JSON only, no markdown"""


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return raw


def analyze_profile_bedrock(image_paths: list[str]) -> dict:
    """Send multiple screenshots to Claude on AWS Bedrock using the Converse API."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # Build content blocks: one image per screenshot + a text prompt
    content = []
    for idx, path in enumerate(image_paths):
        with open(path, "rb") as f:
            image_bytes = f.read()

        if path.endswith(".png"):
            media_format = "png"
        else:
            media_format = "jpeg"

        content.append({
            "text": f"Screenshot {idx + 1} of {len(image_paths)}:"
        })
        content.append({
            "image": {
                "format": media_format,
                "source": {"bytes": image_bytes},
            }
        })

    content.append({
        "text": f"Above are {len(image_paths)} screenshots of the same Hinge profile, scrolled top to bottom. Analyze the FULL profile and generate a response."
    })

    response = client.converse(
        modelId=AWS_BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.8},
    )

    raw = response["output"]["message"]["content"][0]["text"]
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile_openai(image_paths: list[str]) -> dict:
    """Send multiple screenshots to GPT-4o Vision."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    content = []
    for idx, path in enumerate(image_paths):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "text", "text": f"Screenshot {idx + 1} of {len(image_paths)}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })

    content.append({
        "type": "text",
        "text": f"Above are {len(image_paths)} screenshots of the same Hinge profile. Analyze the FULL profile and generate a response.",
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=400,
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile_gemini(image_paths: list[str]) -> dict:
    """Send multiple screenshots to Gemini 1.5 Flash Vision."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    parts = [SYSTEM_PROMPT + "\n\n"]
    for idx, path in enumerate(image_paths):
        parts.append(f"Screenshot {idx + 1} of {len(image_paths)}:")
        parts.append(Image.open(path))

    parts.append(f"Above are {len(image_paths)} screenshots of the same Hinge profile. Analyze the FULL profile and generate a response.")

    response = model.generate_content(
        parts,
        generation_config=genai.GenerationConfig(temperature=0.8, max_output_tokens=400),
    )

    raw = response.text.strip()
    return json.loads(_strip_markdown_fences(raw))


def analyze_profile(image_paths: list[str]) -> dict:
    """Route to configured LLM provider. Accepts a list of screenshot paths."""
    if LLM_PROVIDER == "bedrock":
        return analyze_profile_bedrock(image_paths)
    elif LLM_PROVIDER == "gemini":
        return analyze_profile_gemini(image_paths)
    elif LLM_PROVIDER == "openai":
        return analyze_profile_openai(image_paths)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use bedrock/openai/gemini.")