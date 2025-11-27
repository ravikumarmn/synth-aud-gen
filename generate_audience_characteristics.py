#!/usr/bin/env python3
"""
Audience Characteristics Generator

This script takes attribute slots (from attribute_slots.json) and generates
detailed characteristics for each slot by using the persona template and
slot attributes as input to Gemini LLM.

Input: attribute_slots.json with structure:
  - personas[]: Array of persona data
    - persona_template: Base persona information
    - attributes[]: Array of attribute combinations (slots)

For each attribute slot, it generates:
- about: Behavioral description (interests, digital habits, lifestyle)
- goalsAndMotivations: List of achievement/growth/impact goals
- frustrations: List of process/quality/access challenges
- needState: Current psychological/motivational state
- occasions: Contextual situations for content engagement
"""

import json
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


# Provider configuration
PROVIDER_AZURE = "azure"
PROVIDER_GEMINI = "gemini"


@dataclass
class GeneratedCharacteristics:
    """Generated characteristics for an audience member."""

    member_id: str
    audience_index: int
    about: str
    goals_and_motivations: list[str]
    frustrations: list[str]
    need_state: str
    occasions: str


GENERATION_SYSTEM_PROMPT = """You are an expert persona generator creating realistic, detailed audience member profiles.

Your task is to generate a complete, coherent persona based on:
1. A base persona template (provides personality, goals, frustrations, context)
2. Specific demographic attributes (age group, gender, income, job title, etc.)
3. Screener responses (qualifying criteria)

Generate a realistic, believable individual that:
- Matches all the provided demographic attributes exactly
- Embodies the spirit and characteristics of the base persona
- Has internally consistent traits and behaviors
- Feels like a real person, not a stereotype

Respond ONLY with valid JSON in this exact format:
{{
    "about": "Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle preferences without any demographic markers",
    "goalsAndMotivations": [
        "Achievement-oriented goal focusing on skills or outcomes",
        "Growth-oriented motivation related to learning or development", 
        "Impact-oriented aspiration about influence or contribution"
    ],
    "frustrations": [
        "Process-related challenge about workflows or systems",
        "Quality-related concern about standards or expectations",
        "Access-related barrier about resources or opportunities"
    ],
    "needState": "Current psychological or motivational state expressed in behavioral terms",
    "occasions": "Contextual situations and timing patterns for content engagement, described through activities and behaviors"
}}"""


def create_generation_prompt(member: dict[str, Any]) -> str:
    """
    Create the generation prompt for a single audience member.

    Args:
        member: Audience member dictionary with attributes, persona_template, and screener_responses

    Returns:
        Formatted prompt string for characteristic generation
    """
    persona = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])

    # Format screener Q&A
    screener_section = ""
    if screener_responses:
        screener_lines = []
        for response in screener_responses:
            question = response.get("question", "N/A")
            answer = response.get("answer", "N/A")
            screener_lines.append(f"- **Q**: {question}\n  **A**: {answer}")
        screener_section = "\n".join(screener_lines)
    else:
        screener_section = "No screener responses available."

    prompt = f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}

Above information is enough to understand persona's traits and behavior. Use the screener responses below to create variations and generate a complete, realistic audience member profile as JSON.

## Screener Responses
{screener_section}

## Important Guidelines
1. Use the screener responses to inform lifestyle, work environment, and behavioral descriptions
2. Ensure the generated profile is consistent with the screener answers
3. The profile should feel like a real person, not a stereotype
4. Maintain the spirit of the base persona while adapting to the screener context

Generate a complete, realistic audience member profile as JSON."""

    return prompt


def _create_azure_client() -> tuple[AzureOpenAI, str] | None:
    """
    Create Azure OpenAI client if credentials are available.

    Returns:
        Tuple of (client, model_name) or None if not configured
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key or not endpoint:
        return None

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return client, deployment


def _create_gemini_client() -> tuple[OpenAI, str] | None:
    """
    Create Gemini client if credentials are available.

    Returns:
        Tuple of (client, model_name) or None if not configured
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return client, "gemini-2.5-flash"


def _call_llm(
    client: OpenAI | AzureOpenAI,
    model: str,
    prompt: str,
) -> str:
    """
    Make an LLM API call and return the response content.

    Args:
        client: OpenAI or AzureOpenAI client
        model: Model/deployment name
        prompt: User prompt

    Returns:
        Response content string

    Raises:
        ValueError: If API returns no content
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=4096,
    )

    if not response.choices:
        raise ValueError("API returned no choices")

    message = response.choices[0].message
    content = message.content

    if content is None:
        content = getattr(message, "text", None)
    if content is None:
        finish_reason = response.choices[0].finish_reason
        raise ValueError(f"API returned empty content (finish_reason: {finish_reason})")

    return content.strip()


def _parse_llm_response(
    content: str, member: dict[str, Any]
) -> GeneratedCharacteristics:
    """
    Parse LLM response content into GeneratedCharacteristics.

    Args:
        content: Raw response content from LLM
        member: Original member data for ID extraction

    Returns:
        GeneratedCharacteristics object
    """
    # Handle potential markdown code blocks
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    result_data = json.loads(content)

    return GeneratedCharacteristics(
        member_id=member.get("member_id", "unknown"),
        audience_index=member.get("audience_index", -1),
        about=result_data.get("about", ""),
        goals_and_motivations=result_data.get("goalsAndMotivations", []),
        frustrations=result_data.get("frustrations", []),
        need_state=result_data.get("needState", ""),
        occasions=result_data.get("occasions", ""),
    )


def generate_member_characteristics(
    primary_client: OpenAI | AzureOpenAI,
    primary_model: str,
    member: dict[str, Any],
    fallback_client: OpenAI | AzureOpenAI | None = None,
    fallback_model: str | None = None,
    max_retries: int = 3,
) -> GeneratedCharacteristics | None:
    """
    Generate characteristics for a single audience member using the LLM.
    Uses primary provider first, falls back to secondary on failure.

    Args:
        primary_client: Primary LLM client (Azure OpenAI)
        primary_model: Primary model/deployment name
        member: Audience member to generate characteristics for
        fallback_client: Fallback LLM client (Gemini)
        fallback_model: Fallback model name
        max_retries: Number of retries per provider

    Returns:
        GeneratedCharacteristics or None if all providers fail
    """
    prompt = create_generation_prompt(member)
    providers = [(primary_client, primary_model, "primary")]
    if fallback_client and fallback_model:
        providers.append((fallback_client, fallback_model, "fallback"))

    for client, model, provider_name in providers:
        for attempt in range(max_retries):
            try:
                content = _call_llm(client, model, prompt)
                return _parse_llm_response(content, member)

            except json.JSONDecodeError as e:
                print(
                    f"  JSON parse error ({provider_name}, attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"  API error ({provider_name}, attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        # If primary exhausted retries, try fallback
        if provider_name == "primary" and fallback_client:
            print(f"  Primary provider failed, switching to fallback...")

    return None


def convert_slots_to_members(persona_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert attribute slots to member format expected by generation functions.

    Args:
        persona_data: Persona dictionary with persona_template and attributes list

    Returns:
        List of member dictionaries ready for characteristic generation
    """
    persona_template = persona_data.get("persona_template", {})
    attributes_list = persona_data.get("attributes", [])
    audience_index = persona_data.get("audience_index", 0)

    members = []
    for idx, attributes in enumerate(attributes_list):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "attributes": attributes,
            "persona_template": persona_template,
        }
        members.append(member)

    return members


def generate_audience_characteristics(
    primary_client: OpenAI | AzureOpenAI,
    primary_model: str,
    persona_data: dict[str, Any],
    fallback_client: OpenAI | AzureOpenAI | None = None,
    fallback_model: str | None = None,
    max_workers: int = 5,
) -> dict[str, Any]:
    """
    Generate characteristics for all members in a persona/audience.

    Args:
        primary_client: Primary LLM client (Azure OpenAI)
        primary_model: Primary model/deployment name
        persona_data: Persona dictionary with persona_template and attributes
        fallback_client: Fallback LLM client (Gemini)
        fallback_model: Fallback model name
        max_workers: Maximum number of concurrent API calls

    Returns:
        Dictionary with audience info and generated characteristics
    """
    # Convert slots to members format
    members = convert_slots_to_members(persona_data)
    audience_index = persona_data.get("audience_index", 0)

    print(
        f"\nGenerating characteristics for Audience {audience_index} "
        f"({len(members)} members) with {max_workers} workers..."
    )

    generated_members: list[dict[str, Any]] = []
    failed_members: list[str] = []

    def generate_with_index(
        args: tuple[int, dict],
    ) -> tuple[int, GeneratedCharacteristics | None, dict]:
        idx, member = args
        result = generate_member_characteristics(
            primary_client, primary_model, member, fallback_client, fallback_model
        )
        return idx, result, member

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_with_index, (i, member)): i
            for i, member in enumerate(members)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            idx, result, original_member = future.result()
            member_id = original_member.get("member_id", "unknown")
            print(f"  [{completed}/{len(members)}] Generated: {member_id}")

            if result:
                # Combine original member data with generated characteristics
                enriched_member = {
                    **original_member,
                    "generated_characteristics": asdict(result),
                }
                generated_members.append(enriched_member)
            else:
                # Keep original member even if generation failed
                enriched_member = {
                    **original_member,
                    "generated_characteristics": None,
                    "generation_error": "Failed to generate characteristics",
                }
                generated_members.append(enriched_member)
                failed_members.append(member_id)

    # Sort by member_id to maintain order
    generated_members.sort(key=lambda m: m.get("member_id", ""))

    if failed_members:
        print(f"  Failed to generate for: {', '.join(failed_members)}")

    return {
        "audience_index": audience_index,
        "sample_size": persona_data.get("sample_size", 0),
        "total_slots": persona_data.get("total_slots", 0),
        "persona_template": persona_data.get("persona_template", {}),
        "members": generated_members,
        "generation_stats": {
            "total_members": len(members),
            "successfully_generated": len(members) - len(failed_members),
            "failed": len(failed_members),
        },
    }


def run_generation(
    input_path: Path,
    output_path: Path,
    max_workers: int = 5,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file.
    Uses Azure OpenAI as primary provider and Gemini as fallback.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_workers: Maximum concurrent API calls

    Returns:
        Complete results dictionary with generated characteristics
    """
    # Initialize primary client (Azure OpenAI)
    azure_result = _create_azure_client()
    gemini_result = _create_gemini_client()

    if azure_result:
        primary_client, primary_model = azure_result
        primary_provider = PROVIDER_AZURE
        print(f"Primary provider: Azure OpenAI (deployment: {primary_model})")
    elif gemini_result:
        # Fall back to Gemini as primary if Azure not configured
        primary_client, primary_model = gemini_result
        primary_provider = PROVIDER_GEMINI
        print(f"Primary provider: Gemini (model: {primary_model})")
    else:
        raise ValueError(
            "No LLM provider configured. Set either:\n"
            "  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for Azure OpenAI\n"
            "  - GEMINI_API_KEY for Gemini"
        )

    # Initialize fallback client (Gemini if Azure is primary)
    fallback_client = None
    fallback_model = None
    if primary_provider == PROVIDER_AZURE and gemini_result:
        fallback_client, fallback_model = gemini_result
        print(f"Fallback provider: Gemini (model: {fallback_model})")
    elif primary_provider == PROVIDER_GEMINI:
        print("Fallback provider: None (Gemini is primary)")

    # Load input data (attribute_slots.json format)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate total slots across all personas
    total_slots = data.get(
        "total_slots", sum(p.get("total_slots", 0) for p in data.get("personas", []))
    )

    print(f"Loaded {data.get('total_audiences', 0)} audiences from {input_path}")
    print(f"Total slots to process: {total_slots}")

    # Generate characteristics for each persona
    enriched_audiences: list[dict[str, Any]] = []

    for persona_data in data.get("personas", []):
        enriched_audience = generate_audience_characteristics(
            primary_client,
            primary_model,
            persona_data,
            fallback_client,
            fallback_model,
            max_workers,
        )
        enriched_audiences.append(enriched_audience)

    # Compile results
    total_generated = sum(
        aud["generation_stats"]["successfully_generated"] for aud in enriched_audiences
    )
    total_failed = sum(aud["generation_stats"]["failed"] for aud in enriched_audiences)

    model_info = f"{primary_provider}:{primary_model}"
    if fallback_model:
        model_info += f" (fallback: gemini:{fallback_model})"

    results = {
        "project_name": data.get("project_name"),
        "project_description": data.get("project_description"),
        "project_id": data.get("project_id"),
        "user_id": data.get("user_id"),
        "request_id": data.get("request_id"),
        "generation_model": model_info,
        "primary_provider": primary_provider,
        "fallback_provider": PROVIDER_GEMINI if fallback_client else None,
        "total_audiences": len(enriched_audiences),
        "total_members_processed": total_generated + total_failed,
        "total_successfully_generated": total_generated,
        "total_failed": total_failed,
        "audiences": enriched_audiences,
    }

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of generation results."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Project: {results.get('project_name')}")
    print(f"Model: {results.get('generation_model')}")
    print(f"Total Members Processed: {results.get('total_members_processed')}")
    print(f"Successfully Generated: {results.get('total_successfully_generated')}")
    print(f"Failed: {results.get('total_failed')}")
    print("-" * 60)

    for aud in results.get("audiences", []):
        stats = aud.get("generation_stats", {})
        print(f"\nAudience {aud['audience_index']}:")
        print(f"  Total Members: {stats.get('total_members', 0)}")
        print(f"  Generated: {stats.get('successfully_generated', 0)}")
        print(f"  Failed: {stats.get('failed', 0)}")

        # Show a sample generated member
        members = aud.get("members", [])
        for member in members:
            chars = member.get("generated_characteristics")
            if chars:
                print(f"\n  Sample Generated Member ({member.get('member_id')}):")
                about = chars.get("about", "")
                if len(about) > 150:
                    about = about[:150] + "..."
                print(f"    About: {about}")
                print(f"    Need State: {chars.get('need_state')}")
                goals = chars.get("goals_and_motivations", [])
                if goals:
                    print(f"    Goals: {goals[0]}...")
                break


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate detailed audience characteristics using persona and attributes"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/attribute_slots.json"),
        help="Path to attribute slots JSON file (default: data/attribute_slots.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/audience_characteristics_small.json"),
        help="Path to output file (default: data/audience_characteristics_small.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Maximum concurrent API calls (default: 5)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print("Starting characteristic generation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Workers: {args.workers}")

    try:
        results = run_generation(
            args.input,
            args.output,
            args.workers,
        )
        print_summary(results)
        print(f"\nFull results written to: {args.output}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
