#!/usr/bin/env python3
"""
Audience Characteristics Generator

This script generates detailed audience member characteristics using persona
templates and screener questions as input to Azure OpenAI LLM.

Input: personas_input JSON with structure:
  - audiences[]: Array of audience data
    - persona: Base persona information
    - screenerQuestions[]: Array of qualifying Q&A pairs
    - sampleSize: Number of members to generate

For each audience member, it generates:
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
import asyncio
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()


# Provider configuration
PROVIDER_AZURE = "azure"


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


def _create_azure_client() -> tuple[AsyncAzureOpenAI, str]:
    """
    Create Azure OpenAI async client.

    Returns:
        Tuple of (client, deployment_name)

    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI not configured. Set the following environment variables:\n"
            "  - AZURE_OPENAI_API_KEY\n"
            "  - AZURE_OPENAI_ENDPOINT\n"
            "  - AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to gpt-4o)\n"
            "  - OPENAI_API_VERSION (optional)"
        )

    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return client, deployment


async def _call_llm(
    client: AsyncAzureOpenAI,
    deployment: str,
    prompt: str,
) -> str:
    """
    Make an async LLM API call and return the response content.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        prompt: User prompt

    Returns:
        Response content string

    Raises:
        ValueError: If API returns no content
    """
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=4096,
    )

    if not response.choices:
        raise ValueError("API returned no choices")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("API returned empty content")
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
    # Strip whitespace
    content = content.strip()

    # Handle markdown code blocks (```json ... ``` or ``` ... ```)
    if "```" in content:
        # Extract content between first ``` and last ```
        parts = content.split("```")
        if len(parts) >= 3:
            # Content is between first and second ```
            code_block = parts[1]
            # Remove language identifier if present (e.g., "json\n")
            if code_block.startswith("json"):
                code_block = code_block[4:]
            content = code_block.strip()
        elif len(parts) == 2:
            # Only opening ``` found, take content after it
            code_block = parts[1]
            if code_block.startswith("json"):
                code_block = code_block[4:]
            content = code_block.strip()

    # Try to extract JSON object if there's extra text around it
    if not content.startswith("{"):
        # Find the first { and last }
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx : end_idx + 1]

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


async def generate_member_characteristics(
    client: AsyncAzureOpenAI,
    deployment: str,
    member: dict[str, Any],
    max_retries: int = 3,
) -> GeneratedCharacteristics | None:
    """
    Generate characteristics for a single audience member using Azure OpenAI.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        member: Audience member to generate characteristics for
        max_retries: Number of retries on failure

    Returns:
        GeneratedCharacteristics or None if all retries fail
    """
    prompt = create_generation_prompt(member)

    for attempt in range(max_retries):
        try:
            content = await _call_llm(client, deployment, prompt)
            return _parse_llm_response(content, member)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            print(
                f"    Raw content preview: {content[:200] if content else 'empty'}..."
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)

    return None


def convert_persona_to_template(persona: dict[str, Any]) -> dict[str, Any]:
    """
    Convert persona object from input format to persona_template format.

    Args:
        persona: Persona object with fields like personaName, about, etc.

    Returns:
        Persona template dictionary with standardized field names
    """
    return {
        "id": persona.get("id"),
        "name": persona.get("personaName", ""),
        "type": persona.get("personaType", ""),
        "gender": persona.get("gender", ""),
        "age": persona.get("age"),
        "location": persona.get("location", ""),
        "ethnicity": persona.get("ethnicity", ""),
        "about": persona.get("about", ""),
        "goals_and_motivations": persona.get("goalsAndMotivations", ""),
        "frustrations": persona.get("frustrations", ""),
        "need_state": persona.get("needState", ""),
        "occasions": persona.get("occasions", ""),
    }


def convert_audience_to_members(
    audience_data: dict[str, Any], audience_index: int
) -> list[dict[str, Any]]:
    """
    Convert audience data to member format expected by generation functions.

    Args:
        audience_data: Audience dictionary with persona, screenerQuestions, and sampleSize
        audience_index: Index of this audience in the input

    Returns:
        List of member dictionaries ready for characteristic generation
    """
    persona = audience_data.get("persona", {})
    persona_template = convert_persona_to_template(persona)
    screener_questions = audience_data.get("screenerQuestions", [])
    sample_size = audience_data.get("sampleSize", 1)

    members = []
    for idx in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "persona_template": persona_template,
            "screener_responses": screener_questions,
        }
        members.append(member)

    return members


async def generate_audience_characteristics(
    client: AsyncAzureOpenAI,
    deployment: str,
    audience_data: dict[str, Any],
    audience_index: int,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Generate characteristics for all members in an audience using async.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        audience_data: Audience dictionary with persona, screenerQuestions, sampleSize
        audience_index: Index of this audience
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        Dictionary with generated_audience and metadata sections
    """
    # Convert audience to members format
    members = convert_audience_to_members(audience_data, audience_index)

    print(
        f"\nGenerating characteristics for Audience {audience_index} "
        f"({len(members)} members) with {max_concurrent} concurrent requests..."
    )

    generated_audience: list[dict[str, Any]] = []
    failed_members: list[str] = []
    completed_count = [0]  # Use list to allow mutation in nested function
    total = len(members)

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(
        member: dict[str, Any],
    ) -> tuple[GeneratedCharacteristics | None, dict[str, Any]]:
        async with semaphore:
            result = await generate_member_characteristics(client, deployment, member)
            completed_count[0] += 1
            member_id = member.get("member_id", "unknown")
            print(f"  [{completed_count[0]}/{total}] Generated: {member_id}")
            return result, member

    # Run all tasks concurrently
    tasks = [generate_with_semaphore(member) for member in members]
    results = await asyncio.gather(*tasks)

    for result, original_member in results:
        member_id = original_member.get("member_id", "unknown")

        if result:
            # Store only the generated characteristics with member_id
            audience_member = {
                "member_id": member_id,
                "about": result.about,
                "goals_and_motivations": result.goals_and_motivations,
                "frustrations": result.frustrations,
                "need_state": result.need_state,
                "occasions": result.occasions,
            }
            generated_audience.append(audience_member)
        else:
            # Keep member with error indication
            audience_member = {
                "member_id": member_id,
                "generation_error": "Failed to generate characteristics",
            }
            generated_audience.append(audience_member)
            failed_members.append(member_id)

    # Sort by member_id to maintain order
    generated_audience.sort(key=lambda m: m.get("member_id", ""))

    if failed_members:
        print(f"  Failed to generate for: {', '.join(failed_members)}")

    persona = audience_data.get("persona", {})
    return {
        "generated_audience": generated_audience,
        "metadata": {
            "audience_index": audience_index,
            "sample_size": audience_data.get("sampleSize", 0),
            "persona": persona,
            "screener_questions": audience_data.get("screenerQuestions", []),
            "generation_stats": {
                "total_members": len(members),
                "successfully_generated": len(members) - len(failed_members),
                "failed": len(failed_members),
            },
        },
    }


async def run_generation_async(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file using async.
    Uses Azure OpenAI.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls

    Returns:
        Complete results dictionary with generated characteristics
    """
    # Initialize Azure OpenAI async client
    client, deployment = _create_azure_client()
    print(f"Provider: Azure OpenAI (deployment: {deployment})")

    # Load input data (personas_input format with audiences array)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audiences = data.get("audiences", [])
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)

    print(f"Loaded {len(audiences)} audiences from {input_path}")
    print(f"Total samples to generate: {total_samples}")

    # Generate characteristics for all audiences concurrently
    tasks = [
        generate_audience_characteristics(
            client, deployment, audience_data, idx, max_concurrent
        )
        for idx, audience_data in enumerate(audiences)
    ]
    enriched_audiences = await asyncio.gather(*tasks)

    # Compile results
    total_generated = sum(
        aud["metadata"]["generation_stats"]["successfully_generated"]
        for aud in enriched_audiences
    )
    total_failed = sum(
        aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
    )

    results = {
        "project_name": data.get("projectName"),
        "project_description": data.get("projectDescription"),
        "project_id": data.get("projectId"),
        "user_id": data.get("userId"),
        "request_id": data.get("requestId"),
        "generation_model": f"{PROVIDER_AZURE}:{deployment}",
        "provider": PROVIDER_AZURE,
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

    # Close the client
    await client.close()

    return results


def run_generation(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_generation_async.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls

    Returns:
        Complete results dictionary with generated characteristics
    """
    return asyncio.run(run_generation_async(input_path, output_path, max_concurrent))


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
        metadata = aud.get("metadata", {})
        stats = metadata.get("generation_stats", {})
        print(f"\nAudience {metadata.get('audience_index')}:")
        print(f"  Persona: {metadata.get('persona', {}).get('personaName', 'N/A')}")
        print(f"  Total Members: {stats.get('total_members', 0)}")
        print(f"  Generated: {stats.get('successfully_generated', 0)}")
        print(f"  Failed: {stats.get('failed', 0)}")

        # Show a sample generated member
        generated_audience = aud.get("generated_audience", [])
        for member in generated_audience:
            if "generation_error" not in member:
                print(f"\n  Sample Generated Member ({member.get('member_id')}):")
                about = member.get("about", "")
                if len(about) > 150:
                    about = about[:150] + "..."
                print(f"    About: {about}")
                print(f"    Need State: {member.get('need_state')}")
                goals = member.get("goals_and_motivations", [])
                if goals:
                    print(f"    Goals: {goals[0]}...")
                break


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate detailed audience characteristics using persona templates "
        "and screener questions as input to Azure OpenAI LLM."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/personas_input_10.json"),
        help="Path to personas input JSON file (default: data/personas_input_10.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/audience_characteristics_small.json"),
        help="Path to output file (default: data/audience_characteristics_small.json)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print("Starting characteristic generation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Concurrent Requests: {args.concurrent}")

    start_time = time.time()

    try:
        results = run_generation(
            args.input,
            args.output,
            args.concurrent,
        )

        elapsed_time = time.time() - start_time

        print_summary(results)
        print(f"\nFull results written to: {args.output}")
        print(f"\n{'=' * 60}")
        print(f"TOTAL TIME: {elapsed_time:.2f} seconds")
        print(f"{'=' * 60}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
