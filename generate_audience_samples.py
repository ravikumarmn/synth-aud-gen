#!/usr/bin/env python3
"""
Audience Sample Generator

This script processes audience configurations and generates individual audience
members based on the specified sample size and variable breakdowns.

The breakdown values in the input are raw counts which are converted to percentages,
then used to calculate the actual sample counts for each category.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Any
from itertools import product


def convert_breakdown_to_percentages(
    breakdown: dict[str, int | float],
) -> dict[str, float]:
    """
    Convert raw breakdown counts to percentages.

    Args:
        breakdown: Dictionary with category names as keys and raw counts as values

    Returns:
        Dictionary with category names as keys and percentage values (0-100)
    """
    total = sum(breakdown.values())
    if total == 0:
        # If all values are 0, distribute equally
        count = len(breakdown)
        return {k: 100.0 / count if count > 0 else 0.0 for k in breakdown}

    return {k: (v / total) * 100 for k, v in breakdown.items()}


def calculate_sample_distribution(
    percentages: dict[str, float], sample_size: int
) -> dict[str, int]:
    """
    Calculate actual sample counts from percentages and total sample size.
    Uses largest remainder method to ensure counts sum to sample_size.

    Args:
        percentages: Dictionary with category names and their percentages
        sample_size: Total number of samples to distribute

    Returns:
        Dictionary with category names and their allocated sample counts
    """
    # Calculate initial allocation (floor values)
    raw_allocation = {k: (v / 100) * sample_size for k, v in percentages.items()}
    floor_allocation = {k: int(v) for k, v in raw_allocation.items()}

    # Calculate remainders for largest remainder method
    remainders = {k: raw_allocation[k] - floor_allocation[k] for k in raw_allocation}

    # Determine how many more samples we need to allocate
    allocated = sum(floor_allocation.values())
    remaining = sample_size - allocated

    # Allocate remaining samples to categories with largest remainders
    sorted_by_remainder = sorted(
        remainders.keys(), key=lambda k: remainders[k], reverse=True
    )

    final_allocation = floor_allocation.copy()
    for i in range(remaining):
        if i < len(sorted_by_remainder):
            final_allocation[sorted_by_remainder[i]] += 1

    return final_allocation


def process_variable(variable: dict[str, Any], sample_size: int) -> dict[str, Any]:
    """
    Process a single variable, converting breakdown to percentages and calculating samples.

    Args:
        variable: Variable dictionary containing variableName and breakdown
        sample_size: Total sample size for the audience

    Returns:
        Processed variable with percentages and sample distribution
    """
    result = {
        "variableName": variable.get("variableName"),
    }

    # Include selectedOption if present
    if "selectedOption" in variable:
        result["selectedOption"] = variable["selectedOption"]

    if "breakdown" in variable:
        breakdown = variable["breakdown"]

        # Convert to percentages
        percentages = convert_breakdown_to_percentages(breakdown)

        # Calculate sample distribution
        sample_distribution = calculate_sample_distribution(percentages, sample_size)

        result["breakdown_percentages"] = {
            k: round(v, 2) for k, v in percentages.items()
        }
        result["sample_distribution"] = sample_distribution
        result["original_breakdown"] = breakdown

    return result


def generate_audience_members(
    variables: list[dict[str, Any]],
    sample_size: int,
    base_persona: dict[str, Any] | None,
    audience_index: int,
) -> list[dict[str, Any]]:
    """
    Generate individual audience members based on variable distributions.

    Args:
        variables: List of processed variables with sample distributions
        sample_size: Total number of audience members to generate
        base_persona: Base persona template to use for each member
        audience_index: Index of the audience

    Returns:
        List of individual audience member dictionaries
    """
    # Build pools of values for each variable based on distribution
    variable_pools: dict[str, list[str]] = {}

    for var in variables:
        var_name = var["variableName"]
        if "sample_distribution" in var:
            pool = []
            for category, count in var["sample_distribution"].items():
                pool.extend([category] * count)
            random.shuffle(pool)
            variable_pools[var_name] = pool

    # Generate individual audience members
    members = []
    for i in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{i + 1:04d}",
            "audience_index": audience_index,
            "attributes": {},
        }

        # Assign values from each variable pool
        for var in variables:
            var_name = var["variableName"]
            if var_name in variable_pools and variable_pools[var_name]:
                # Pop from pool to ensure exact distribution
                member["attributes"][var_name] = variable_pools[var_name].pop()

            # Include selectedOption if present
            if "selectedOption" in var:
                member["attributes"][f"{var_name}_option"] = var["selectedOption"]

        # Include base persona info
        if base_persona:
            member["persona_template"] = {
                "id": base_persona.get("id"),
                "name": base_persona.get("personaName"),
                "type": base_persona.get("personaType"),
                "base_gender": base_persona.get("gender"),
                "base_age": base_persona.get("age"),
                "location": base_persona.get("location"),
                "ethnicity": base_persona.get("ethnicity"),
                "about": base_persona.get("about"),
                "goals_and_motivations": base_persona.get("goalsAndMotivations"),
                "frustrations": base_persona.get("frustrations"),
                "need_state": base_persona.get("needState"),
                "occasions": base_persona.get("occasions"),
            }

        members.append(member)

    return members


def process_audience(audience: dict[str, Any], audience_index: int) -> dict[str, Any]:
    """
    Process a single audience configuration and generate all members.

    Args:
        audience: Audience dictionary with sampleSize and variables
        audience_index: Index of the audience (for identification)

    Returns:
        Processed audience with calculated distributions and generated members
    """
    sample_size = audience.get("sampleSize", 0)
    has_quota = audience.get("hasQuota", False)

    # Process variables first
    processed_variables = []
    for variable in audience.get("variables", []):
        processed_var = process_variable(variable, sample_size)
        processed_variables.append(processed_var)

    # Get base persona
    base_persona = audience.get("persona")

    # Generate all audience members
    members = generate_audience_members(
        processed_variables,
        sample_size,
        base_persona,
        audience_index,
    )

    result = {
        "audience_index": audience_index,
        "sample_size": sample_size,
        "has_quota": has_quota,
        "variable_summary": processed_variables,
        "members": members,
    }

    return result


def process_input_file(input_path: Path) -> dict[str, Any]:
    """
    Process the input JSON file and generate all audience members.

    Args:
        input_path: Path to the input JSON file

    Returns:
        Processed output dictionary with all generated audience members
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = {
        "project_name": data.get("projectName"),
        "project_description": data.get("projectDescription"),
        "project_id": data.get("projectId"),
        "user_id": data.get("userId"),
        "request_id": data.get("requestId"),
        "total_audiences": len(data.get("audiences", [])),
        "audiences": [],
    }

    # Process each audience
    for idx, audience in enumerate(data.get("audiences", [])):
        processed_audience = process_audience(audience, idx)
        output["audiences"].append(processed_audience)

    # Calculate total samples across all audiences
    output["total_members_generated"] = sum(
        len(aud["members"]) for aud in output["audiences"]
    )

    return output


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate audience sample distributions from input configuration"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/persona_input.json"),
        help="Path to input JSON file (default: data/persona_input.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: data/audience_samples_output.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty print JSON output (default: True)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = Path("data/audience_samples_output.json")

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"Processing input file: {args.input}")

    # Process the input
    result = process_input_file(args.input)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(result, f, ensure_ascii=False)

    print(f"Output written to: {args.output}")
    print(f"Total audiences processed: {result['total_audiences']}")
    print(f"Total members generated: {result['total_members_generated']}")

    # Print summary for each audience
    for aud in result["audiences"]:
        print(f"\n--- Audience {aud['audience_index']} ---")
        print(f"  Sample Size: {aud['sample_size']}")
        print(f"  Members Generated: {len(aud['members'])}")
        for var in aud["variable_summary"]:
            print(f"  {var['variableName']}:")
            if "breakdown_percentages" in var:
                for cat, pct in var["breakdown_percentages"].items():
                    samples = var["sample_distribution"].get(cat, 0)
                    print(f"    {cat}: {pct}% -> {samples} samples")


if __name__ == "__main__":
    main()
