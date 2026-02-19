# services/explainer.py

import os
from openai import OpenAI

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cache = {}

def rule_based_explanation(label: str, confidence: float) -> str:
    confidence_pct = round(confidence * 100, 2)

    explanations = {
        "Dog 🐶": (
            f"This appears to be a dog with {confidence_pct}% confidence. "
            "Dogs are domesticated animals known for loyalty, companionship, and intelligence."
        ),
        "Cat 🐱": (
            f"This appears to be a cat with {confidence_pct}% confidence. "
            "Cats are independent, curious, and commonly kept as household pets."
        ),
        "Unknown": (
            "The model is not confident about this image. Please try a clearer image."
        )
    }

    return explanations.get(label, explanations["Unknown"])


def llm_explanation(label: str, confidence: float) -> str:
    confidence_pct = round(confidence * 100, 2)

    prompt = f"""
    You are an AI vision assistant.

    The detected object is: {label}
    Confidence score: {confidence_pct}%

    Provide a short, professional explanation of this detection.
    Mention the confidence level naturally.
    Keep it under 100 words.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain computer vision outputs clearly and professionally."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


def generate_explanation(label: str, confidence: float) -> str:
    key = f"{label}_{round(confidence,1)}"
    if key in cache:
        return cache[key]
    try:
        if client:
            explanation = llm_explanation(label, confidence)
        else:
            explanation = rule_based_explanation(label, confidence)

        cache[key] = explanation
        return explanation

    except:
        return rule_based_explanation(label, confidence)
