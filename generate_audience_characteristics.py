#!/usr/bin/env python3
"""
Audience Characteristics Generator

This script takes generated audience samples and enriches each member with
detailed characteristics by using the persona template and member attributes
as input to Gemini LLM.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
        member: Audience member dictionary with attributes and persona_template

    Returns:
        Formatted prompt string for characteristic generation
    """
    persona = member.get("persona_template", {})

    prompt = f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}

Above information is enough to understand persona's traits and behavior. But you have to use audience initially provided attributes to generate a complete, realistic audience member profile as JSON.

## Audience Attributes
- **Age Group**: {member.get('attributes', {}).get('Age Group', 'N/A')}
- **Income**: {member.get('attributes', {}).get('Income', 'N/A')}
- **Employees in company**: {member.get('attributes', {}).get('Employees in company', 'N/A')}
- **Job Title**: {member.get('attributes', {}).get('Job Title', 'N/A')}

## Important Guidelines
1. The generated name should be culturally appropriate for the ethnicity
2. Age must fall within the specified age group range
3. Gender must match the 'Age Group' attribute (male/female) if specified
4. Job title/occupation must align with the 'Job Title' attribute
5. Income level should influence lifestyle descriptions
6. Company size should influence work environment descriptions

Generate a complete, realistic audience member profile as JSON."""

    return prompt


def generate_member_characteristics(
    client: OpenAI,
    member: dict[str, Any],
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> GeneratedCharacteristics | None:
    """
    Generate characteristics for a single audience member using the LLM.

    Args:
        client: OpenAI client configured for Gemini
        member: Audience member to generate characteristics for
        model: Model name to use
        max_retries: Number of retries on failure

    Returns:
        GeneratedCharacteristics or None if generation fails
    """
    prompt = create_generation_prompt(member)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,  # Higher temperature for creative generation
                max_tokens=4096,  # Increased to avoid truncation
            )

            if not response.choices:
                raise ValueError("API returned no choices")

            message = response.choices[0].message
            content = message.content

            if content is None:
                content = getattr(message, "text", None)
            if content is None:
                finish_reason = response.choices[0].finish_reason
                raise ValueError(
                    f"API returned empty content (finish_reason: {finish_reason})"
                )
            content = content.strip()

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

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def generate_audience_characteristics(
    client: OpenAI,
    audience: dict[str, Any],
    model: str = "gemini-2.5-flash",
    max_workers: int = 5,
) -> dict[str, Any]:
    """
    Generate characteristics for all members in an audience.

    Args:
        client: OpenAI client configured for Gemini
        audience: Audience dictionary with members
        model: Model name to use
        max_workers: Maximum number of concurrent API calls

    Returns:
        Dictionary with audience info and generated characteristics
    """
    members = audience.get("members", [])
    audience_index = audience.get("audience_index", 0)

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
        return idx, generate_member_characteristics(client, member, model), member

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
        "sample_size": audience.get("sample_size", 0),
        "has_quota": audience.get("has_quota", False),
        "variable_summary": audience.get("variable_summary", []),
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
    model: str = "gemini-2.5-flash",
    max_workers: int = 5,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        model: Model name to use
        max_workers: Maximum concurrent API calls

    Returns:
        Complete results dictionary with generated characteristics
    """
    # Initialize OpenAI client with Gemini endpoint
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. "
            "Get your API key from https://aistudio.google.com/apikey"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {data.get('total_audiences', 0)} audiences from {input_path}")
    print(f"Total members to process: {data.get('total_members_generated', 0)}")

    # Generate characteristics for each audience
    enriched_audiences: list[dict[str, Any]] = []

    for audience in data.get("audiences", []):
        enriched_audience = generate_audience_characteristics(
            client, audience, model, max_workers
        )
        enriched_audiences.append(enriched_audience)

    # Compile results
    total_generated = sum(
        aud["generation_stats"]["successfully_generated"] for aud in enriched_audiences
    )
    total_failed = sum(aud["generation_stats"]["failed"] for aud in enriched_audiences)

    results = {
        "project_name": data.get("project_name"),
        "project_description": data.get("project_description"),
        "project_id": data.get("project_id"),
        "user_id": data.get("user_id"),
        "request_id": data.get("request_id"),
        "generation_model": model,
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
        default=Path("data/audience_samples_small.json"),
        help="Path to audience samples JSON file (default: data/audience_samples_small.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/audience_characteristics_small.json"),
        help="Path to output file (default: data/audience_characteristics_small.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for generation (default: gemini-2.5-flash)",
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

    print(f"Starting characteristic generation with {args.model}...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Workers: {args.workers}")

    try:
        results = run_generation(
            args.input,
            args.output,
            args.model,
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
