#!/usr/bin/env python3
"""
Audience-Persona Alignment Evaluator

Uses Azure OpenAI GPT-4o as an LLM judge to evaluate how well generated audience
members align with their source persona characteristics. Scores each member
from 1-5 based on coherence and plausibility.
"""

import json
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvaluationResult:
    """Result of evaluating a single audience member."""

    member_id: str
    audience_index: int
    alignment_score: int  # 1-5
    reasoning: str
    strengths: list[str]
    concerns: list[str]


@dataclass
class AudienceEvaluationSummary:
    """Summary of evaluation for an entire audience."""

    audience_index: int
    total_members_evaluated: int
    average_score: float
    score_distribution: dict[int, int]
    sample_evaluations: list[EvaluationResult]


EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator assessing the alignment between generated audience members and their source persona.

Your task is to evaluate whether an audience member's attributes are coherent and plausible given the base persona characteristics.

Scoring Criteria (1-5):
- **5 (Excellent)**: Attributes are highly coherent with persona. Demographics, professional details, and context align naturally.
- **4 (Good)**: Attributes mostly align with persona. Minor inconsistencies that don't break plausibility.
- **3 (Acceptable)**: Attributes are somewhat plausible but have noticeable gaps or tensions with persona.
- **2 (Poor)**: Significant misalignment between attributes and persona. Hard to imagine this person realistically.
- **1 (Very Poor)**: Attributes fundamentally contradict the persona. Completely implausible combination.

Consider these factors:
1. **Demographic Coherence**: Do age, gender, income align with persona's described lifestyle?
2. **Professional Plausibility**: Does job title/company size fit the persona's background and career stage?
3. **Internal Consistency**: Do all attributes work together as a believable person?

Respond ONLY with valid JSON in this exact format:
{
    "alignment_score": <1-5>,
    "reasoning": "<2-3 sentence explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "concerns": ["<concern 1>", "<concern 2>"]
}"""


def create_evaluation_prompt(member: dict[str, Any]) -> str:
    """
    Create the evaluation prompt for a single audience member.

    Args:
        member: Audience member dictionary with attributes and persona_template

    Returns:
        Formatted prompt string for evaluation
    """
    persona = member.get("persona_template", {})
    attributes = member.get("attributes", {})

    prompt = f"""Evaluate the following audience member against their source persona:

## Source Persona
- **Name**: {persona.get('name', 'N/A')}
- **Type**: {persona.get('type', 'N/A')}
- **Base Age**: {persona.get('base_age', 'N/A')}
- **Base Gender**: {persona.get('base_gender', 'N/A')}
- **Location**: {persona.get('location', 'N/A')}
- **Ethnicity**: {persona.get('ethnicity', 'N/A')}
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}

## Generated Audience Member Attributes
"""

    for key, value in attributes.items():
        prompt += f"- **{key}**: {value}\n"

    prompt += "\nProvide your evaluation as JSON."

    return prompt


def evaluate_member(
    client: AzureOpenAI,
    member: dict[str, Any],
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> EvaluationResult | None:
    """
    Evaluate a single audience member using the LLM judge.

    Args:
        client: AzureOpenAI client
        member: Audience member to evaluate
        model: Model name (deployment name) to use
        max_retries: Number of retries on failure

    Returns:
        EvaluationResult or None if evaluation fails
    """
    prompt = create_evaluation_prompt(member)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent scoring
                max_tokens=2048,
            )

            # Debug: check response structure
            if not response.choices:
                raise ValueError("API returned no choices")

            message = response.choices[0].message
            content = message.content

            if content is None:
                # Try alternative fields
                content = getattr(message, "text", None)
            if content is None:
                # Check if there's a refusal or finish reason issue
                finish_reason = response.choices[0].finish_reason
                raise ValueError(
                    f"API returned empty content (finish_reason: {finish_reason})"
                )
            content = content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result_data = json.loads(content)

            return EvaluationResult(
                member_id=member.get("member_id", "unknown"),
                audience_index=member.get("audience_index", -1),
                alignment_score=int(result_data.get("alignment_score", 3)),
                reasoning=result_data.get("reasoning", ""),
                strengths=result_data.get("strengths", []),
                concerns=result_data.get("concerns", []),
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


def evaluate_audience(
    client: AzureOpenAI,
    audience: dict[str, Any],
    model: str = "gpt-4o",
    max_workers: int = 5,
) -> AudienceEvaluationSummary:
    """
    Evaluate all members from an audience using parallel API calls.

    Args:
        client: AzureOpenAI client
        audience: Audience dictionary with members
        model: Model name (deployment name) to use
        max_workers: Maximum number of concurrent API calls

    Returns:
        AudienceEvaluationSummary with results
    """
    members = audience.get("members", [])
    audience_index = audience.get("audience_index", 0)

    print(
        f"\nEvaluating Audience {audience_index} ({len(members)} members) with {max_workers} workers..."
    )

    evaluations: list[EvaluationResult] = []
    score_distribution: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    failed_members: list[str] = []

    def evaluate_with_index(
        args: tuple[int, dict],
    ) -> tuple[int, EvaluationResult | None]:
        idx, member = args
        return idx, evaluate_member(client, member, model)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_with_index, (i, member)): i
            for i, member in enumerate(members)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            idx, result = future.result()
            member_id = members[idx].get("member_id", "unknown")
            print(f"  [{completed}/{len(members)}] Evaluated: {member_id}")

            if result:
                evaluations.append(result)
                score_distribution[result.alignment_score] += 1
            else:
                failed_members.append(member_id)

    if failed_members:
        print(f"  Failed to evaluate: {', '.join(failed_members)}")

    # Calculate average score
    if evaluations:
        avg_score = sum(e.alignment_score for e in evaluations) / len(evaluations)
    else:
        avg_score = 0.0

    return AudienceEvaluationSummary(
        audience_index=audience_index,
        total_members_evaluated=len(evaluations),
        average_score=round(avg_score, 2),
        score_distribution=score_distribution,
        sample_evaluations=evaluations,
    )


def run_evaluation(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """
    Run evaluation on all audiences in the input file.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write evaluation results
        model: Model name (deployment name) to use

    Returns:
        Complete evaluation results dictionary
    """
    # Initialize Azure OpenAI client
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required.")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {data.get('total_audiences', 0)} audiences from {input_path}")

    # Evaluate each audience
    audience_summaries: list[AudienceEvaluationSummary] = []

    for audience in data.get("audiences", []):
        summary = evaluate_audience(client, audience, model)
        audience_summaries.append(summary)

    # Compile results
    results = {
        "project_name": data.get("project_name"),
        "project_id": data.get("project_id"),
        "evaluation_model": model,
        "total_audiences_evaluated": len(audience_summaries),
        "overall_average_score": round(
            (
                sum(s.average_score for s in audience_summaries)
                / len(audience_summaries)
                if audience_summaries
                else 0
            ),
            2,
        ),
        "audience_evaluations": [asdict(s) for s in audience_summaries],
    }

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Project: {results.get('project_name')}")
    print(f"Model: {results.get('evaluation_model')}")
    print(f"Overall Average Score: {results.get('overall_average_score')}/5")
    print("-" * 60)

    for aud_eval in results.get("audience_evaluations", []):
        print(f"\nAudience {aud_eval['audience_index']}:")
        print(f"  Members Evaluated: {aud_eval['total_members_evaluated']}")
        print(f"  Average Score: {aud_eval['average_score']}/5")
        print("  Score Distribution:")
        for score, count in sorted(aud_eval["score_distribution"].items()):
            bar = "█" * count
            print(f"    {score}: {bar} ({count})")

        # Show a sample evaluation
        if aud_eval["sample_evaluations"]:
            sample = aud_eval["sample_evaluations"][0]
            print(f"\n  Sample Evaluation ({sample['member_id']}):")
            print(f"    Score: {sample['alignment_score']}/5")
            print(f"    Reasoning: {sample['reasoning']}")


def main() -> None:
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate audience-persona alignment using LLM-as-judge"
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
        default=Path("data/evaluation_results_small.json"),
        help="Path to output evaluation results (default: data/evaluation_results_small.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model (deployment name) to use for evaluation (default: gpt-4o)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"Starting evaluation with {args.model}...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    try:
        results = run_evaluation(
            args.input,
            args.output,
            args.model,
        )
        print_summary(results)
        print(f"\nFull results written to: {args.output}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
