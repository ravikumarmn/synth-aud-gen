#!/usr/bin/env python3
"""
Audience Sample Generator

This script processes audience configurations and generates individual audience
members based on the specified sample size and variable breakdowns.

The breakdown values in the input are raw counts which are converted to percentages,
then used to calculate the actual sample counts for each category.

Key Features:
- Generates all valid attribute combinations (permutations) with exact proportions
- Saves attribute slots to a separate JSON file before generating members
- Ensures exact match of sample size and proportions
"""

import json
import argparse
import random
import math
from pathlib import Path
from typing import Any
from itertools import product
from functools import reduce


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


def compute_attribute_combinations(
    variables: list[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    """
    Compute all valid attribute combinations with exact proportions.

    Uses the Cartesian product of all variable categories and calculates
    the expected count for each combination based on the product of
    individual variable proportions.

    Args:
        variables: List of processed variables with sample distributions
        sample_size: Total number of samples to distribute

    Returns:
        List of attribute combination dictionaries with counts
    """
    # Extract variable names and their distributions
    var_data: list[tuple[str, dict[str, float], str | None]] = []

    for var in variables:
        var_name = var["variableName"]
        if "breakdown_percentages" in var:
            # Filter out categories with 0%
            percentages = {
                k: v for k, v in var["breakdown_percentages"].items() if v > 0
            }
            selected_option = var.get("selectedOption")
            var_data.append((var_name, percentages, selected_option))

    if not var_data:
        return []

    # Get all category options for each variable
    var_names = [v[0] for v in var_data]
    var_categories = [list(v[1].keys()) for v in var_data]
    var_percentages = [v[1] for v in var_data]
    var_selected_options = [v[2] for v in var_data]

    # Generate all combinations using Cartesian product
    all_combinations = list(product(*var_categories))

    # Calculate expected count for each combination
    combination_slots: list[dict[str, Any]] = []

    for combo in all_combinations:
        # Calculate combined probability (product of individual percentages)
        combined_percentage = 1.0
        for i, category in enumerate(combo):
            combined_percentage *= var_percentages[i][category] / 100.0

        # Calculate expected count
        expected_count = combined_percentage * sample_size

        # Build attributes dict
        attributes: dict[str, str] = {}
        for i, category in enumerate(combo):
            attributes[var_names[i]] = category
            # Add selected option if present
            if var_selected_options[i]:
                attributes[f"{var_names[i]}_option"] = var_selected_options[i]

        combination_slots.append(
            {
                "attributes": attributes,
                "_percentage": round(combined_percentage * 100, 4),  # Internal use only
                "_count": 0,  # Internal use only, will be filled by allocation
            }
        )

    # Allocate counts using largest remainder method to ensure exact sample_size
    combination_slots = allocate_combination_counts(combination_slots, sample_size)

    return combination_slots


def allocate_combination_counts(
    combinations: list[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    """
    Allocate exact counts to combinations using largest remainder method.

    Args:
        combinations: List of combination dicts with percentage
        sample_size: Total samples to allocate

    Returns:
        Updated combinations with count filled
    """
    # Calculate expected counts and floor values
    expected_counts = [
        (combo["_percentage"] / 100.0) * sample_size for combo in combinations
    ]

    for i, combo in enumerate(combinations):
        combo["_count"] = int(expected_counts[i])

    # Calculate remainders
    remainders = [
        (i, expected_counts[i] - combinations[i]["_count"])
        for i in range(len(combinations))
    ]

    # Sort by remainder descending
    remainders.sort(key=lambda x: x[1], reverse=True)

    # Allocate remaining samples
    allocated = sum(c["_count"] for c in combinations)
    remaining = sample_size - allocated

    for i in range(remaining):
        if i < len(remainders):
            idx = remainders[i][0]
            combinations[idx]["_count"] += 1

    return combinations


def save_attribute_slots(
    slots_data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save attribute slots/combinations to a separate JSON file.

    Args:
        slots_data: Dictionary containing all audience slots
        output_path: Path to save the slots JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(slots_data, f, indent=2, ensure_ascii=False)
    print(f"Attribute slots saved to: {output_path}")


def generate_audience_members(
    variables: list[dict[str, Any]],
    sample_size: int,
    base_persona: dict[str, Any] | None,
    audience_index: int,
    attribute_slots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate individual audience members based on pre-computed attribute slots.

    Uses the pre-computed attribute combinations to ensure exact proportions
    are maintained across all variable combinations.

    Args:
        variables: List of processed variables with sample distributions
        sample_size: Total number of audience members to generate
        base_persona: Base persona template to use for each member
        audience_index: Index of the audience
        attribute_slots: Pre-computed attribute combinations with counts

    Returns:
        List of individual audience member dictionaries
    """
    members: list[dict[str, Any]] = []
    member_counter = 1

    if attribute_slots:
        # Use pre-computed slots for exact proportions
        for slot in attribute_slots:
            count = slot["_count"]
            attributes = slot["attributes"]

            for _ in range(count):
                member = {
                    "member_id": f"AUD{audience_index}_{member_counter:04d}",
                    "audience_index": audience_index,
                    "attributes": attributes.copy(),
                }

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
                        "goals_and_motivations": base_persona.get(
                            "goalsAndMotivations"
                        ),
                        "frustrations": base_persona.get("frustrations"),
                        "need_state": base_persona.get("needState"),
                        "occasions": base_persona.get("occasions"),
                    }

                members.append(member)
                member_counter += 1
    else:
        # Fallback to old pool-based method if no slots provided
        variable_pools: dict[str, list[str]] = {}

        for var in variables:
            var_name = var["variableName"]
            if "sample_distribution" in var:
                pool = []
                for category, count in var["sample_distribution"].items():
                    pool.extend([category] * count)
                random.shuffle(pool)
                variable_pools[var_name] = pool

        for i in range(sample_size):
            member = {
                "member_id": f"AUD{audience_index}_{i + 1:04d}",
                "audience_index": audience_index,
                "attributes": {},
            }

            for var in variables:
                var_name = var["variableName"]
                if var_name in variable_pools and variable_pools[var_name]:
                    member["attributes"][var_name] = variable_pools[var_name].pop()

                if "selectedOption" in var:
                    member["attributes"][f"{var_name}_option"] = var["selectedOption"]

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

    # Shuffle to randomize order while maintaining exact proportions
    random.shuffle(members)

    # Re-assign member IDs after shuffle to maintain sequential order
    for i, member in enumerate(members):
        member["member_id"] = f"AUD{audience_index}_{i + 1:04d}"

    return members


def process_audience(
    audience: dict[str, Any],
    audience_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Process a single audience configuration and generate all members.

    Args:
        audience: Audience dictionary with sampleSize and variables
        audience_index: Index of the audience (for identification)

    Returns:
        Tuple of (processed audience dict, attribute slots list)
    """
    sample_size = audience.get("sampleSize", 0)
    has_quota = audience.get("hasQuota", False)

    # Process variables first
    processed_variables = []
    for variable in audience.get("variables", []):
        processed_var = process_variable(variable, sample_size)
        processed_variables.append(processed_var)

    # Compute all attribute combinations with exact proportions
    attribute_slots = compute_attribute_combinations(processed_variables, sample_size)

    # Get base persona
    base_persona = audience.get("persona")

    # Generate all audience members using pre-computed slots
    members = generate_audience_members(
        processed_variables,
        sample_size,
        base_persona,
        audience_index,
        attribute_slots,  # Pass pre-computed slots
    )

    result = {
        "audience_index": audience_index,
        "sample_size": sample_size,
        "has_quota": has_quota,
        "variable_summary": processed_variables,
        "members": members,
    }

    return result, attribute_slots


def process_input_file(
    input_path: Path,
    slots_output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Process the input JSON file and generate all audience members.

    Saves attribute slots to a separate JSON file before generating members.

    Args:
        input_path: Path to the input JSON file
        slots_output_path: Path to save attribute slots JSON (optional)

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

    # Collect all attribute slots for saving
    all_slots_data: dict[str, Any] = {
        "project_name": output["project_name"],
        "project_id": output["project_id"],
        "request_id": output["request_id"],
        "total_audiences": output["total_audiences"],
        "personas": [],
    }

    # Process each audience
    for idx, audience in enumerate(data.get("audiences", [])):
        processed_audience, attribute_slots = process_audience(audience, idx)
        output["audiences"].append(processed_audience)

        # Get persona template from input
        persona_input = audience.get("persona", {})
        persona_template = {
            "id": persona_input.get("id"),
            "name": persona_input.get("personaName"),
            "type": persona_input.get("personaType"),
            "gender": persona_input.get("gender"),
            "age": persona_input.get("age"),
            "location": persona_input.get("location"),
            "ethnicity": persona_input.get("ethnicity"),
            "about": persona_input.get("about"),
            "goals_and_motivations": persona_input.get("goalsAndMotivations"),
            "frustrations": persona_input.get("frustrations"),
            "need_state": persona_input.get("needState"),
            "occasions": persona_input.get("occasions"),
        }

        # Collect slots data - format: persona_template with its attributes
        # Expand slots to match sample_size (repeat each attribute based on its count)
        expanded_attributes: list[dict[str, str]] = []
        for slot in attribute_slots:
            count = slot["_count"]
            if count > 0:
                for _ in range(count):
                    expanded_attributes.append(slot["attributes"])

        persona_slots = {
            "audience_index": idx,
            "sample_size": processed_audience["sample_size"],
            "total_slots": len(expanded_attributes),
            "persona_template": persona_template,
            "attributes": expanded_attributes,
        }
        all_slots_data["personas"].append(persona_slots)

    # Calculate total samples across all audiences
    output["total_members_generated"] = sum(
        len(aud["members"]) for aud in output["audiences"]
    )

    # Calculate total slots stats
    all_slots_data["total_slots"] = sum(
        p["total_slots"] for p in all_slots_data["personas"]
    )

    # Save attribute slots to separate file if path provided
    if slots_output_path:
        save_attribute_slots(all_slots_data, slots_output_path)

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
        "-s",
        "--slots-output",
        type=Path,
        default=None,
        help="Path to save attribute slots JSON file (default: data/attribute_slots.json)",
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

    # Set default slots output path if not provided
    if args.slots_output is None:
        args.slots_output = Path("data/attribute_slots.json")

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"Processing input file: {args.input}")
    print(f"Attribute slots will be saved to: {args.slots_output}")

    # Process the input and save attribute slots
    result = process_input_file(args.input, args.slots_output)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(result, f, ensure_ascii=False)

    print(f"\nOutput written to: {args.output}")
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
