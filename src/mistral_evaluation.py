import json
from typing import List, Optional, Set
from pydantic import BaseModel, Field
from groq import Groq
import os
import time
from dotenv import load_dotenv
load_dotenv()
# -------------------------------
# Pydantic Models
# -------------------------------

class EvaluationScores(BaseModel):
    length: int 
    extension: int | str
    relevance: int 
    brandability: int
    literalness: int
    style: int
    safety: int

class DomainEvaluation(BaseModel):
    domain: str
    scores: EvaluationScores
    issues: List[str] = []
    comments: str = ""

class EvaluationResult(BaseModel):
    evaluations: List[DomainEvaluation]

class Business(BaseModel):
    fsq_place_id: str
    name: str
    normalized_description: str
    sector: Optional[str] = None
    website: Optional[str] = None
    city: Optional[str] = None
    categories: Optional[List[str]] = None


class Entry(BaseModel):
    business: Business
    parsed_suggestions: List[str] = []
    evaluation: Optional[EvaluationResult] = None

class GemmaDomainSuggestion(BaseModel):
    domain: str
    confidence: float
    reasoning: Optional[str]

class GemmaEntry(BaseModel):
    business: Business
    parsed_suggestions: List[GemmaDomainSuggestion] = []
    evaluation: Optional[EvaluationResult] = None
# -------------------------------
# Groq Client
# -------------------------------

client = Groq(api_key=os.getenv("THIRD_GROQ_API_KEY"))

# -------------------------------
# Prompt Builder
# -------------------------------

def build_prompt(description: str, suggestions: List[str]) -> str:
    return f"""
You are an expert domain name evaluator. 
Your task is to assess whether each suggested domain name is a good fit for a business, according to the following rules:

Evaluation Criteria:
1. Length: 6–15 characters (excluding ".com").
2. Extension: Must end with ".com".
3. Relevance: Clearly related to the business description.
4. Brandability: Memorable, easy to spell and pronounce, not too generic.
5. Literalness: Should not be a verbatim copy of the full business name; shorter, more brandable forms are preferred.
6. Style: Must not contain hyphens, numbers, or special characters.
7. Safety: Must not contain inappropriate, offensive, or misleading content.

For each candidate domain, check all criteria and provide:
- A rating from 1 (very poor) to 5 (excellent) for each criterion.
- A short explanation of any violations or weaknesses.
- A failure category label if it violates a rule, chosen from:
  ["too_long", "too_short", "wrong_extension", "irrelevant", "not_brandable", "too_literal", "bad_style", "unsafe", "other"]

Business description:
"{description}"

Domain suggestions:
{suggestions}

Respond in strict JSON format, with the following structure:
{{
  "evaluations": [
    {{
      "domain": "example.com",
      "scores": {{
        "length": 4,
        "extension": 5,
        "relevance": 5,
        "brandability": 3,
        "literalness": 2,
        "style": 5,
        "safety": 5
      }},
      "issues": ["too_literal"],
      "comments": "Domain is relevant but too long and literal."
    }}
  ]
}}
"""


# -------------------------------
# Evaluation Function
# -------------------------------

def evaluate_domains(description: str, suggestions: List[str]) -> Optional[EvaluationResult]:
    """Send prompt to LLM judge via Groq API and parse result into Pydantic model."""
    prompt = build_prompt(description, suggestions)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        eval_text = response.choices[0].message.content.strip()
        eval_json = json.loads(eval_text)

        return EvaluationResult(**eval_json)

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


# -------------------------------
# Main Pipeline
# -------------------------------

def run_mistral_evaluation(input_file: str, output_file: str, temp_file: str = "temp_results.json") -> None:
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Map all entries by fsq_place_id
    entries_by_id = {entry["business"]["fsq_place_id"]: Entry(**entry) for entry in raw_data}

    seen_ids: Set[str] = set()

    # Resume from checkpoint if temp file exists
    if os.path.exists(temp_file):
        with open(temp_file, "r", encoding="utf-8") as f:
            temp_data = json.load(f)

        # Update entries_by_id with already processed results
        for entry_data in temp_data:
            e = Entry(**entry_data)
            entries_by_id[e.business.fsq_place_id] = e
            seen_ids.add(e.business.fsq_place_id)

        print(f"🔄 Resuming from checkpoint: {len(seen_ids)} entries already processed")

    # Process remaining entries
    for i, entry in enumerate(entries_by_id.values(), start=1):
        if entry.business.fsq_place_id in seen_ids:
            print(f"⏩ Skipping entry {i} (already processed): {entry.business.name}")
            continue

        print(f"\n🔎 Evaluating entry {i}: {entry.business.name}")

        if not entry.business.normalized_description or not entry.parsed_suggestions:
            print("  ⚠️ Skipped (missing description or suggestions)")
        else:
            evaluation = evaluate_domains(entry.business.normalized_description, entry.parsed_suggestions)
            if evaluation:
                entry.evaluation = evaluation
                print(f"  ✅ Done: {len(evaluation.evaluations)} domains evaluated")
            else:
                print("  ❌ No evaluation returned")

        seen_ids.add(entry.business.fsq_place_id)

        # Update checkpoint (save only processed entries)
        with open(temp_file, "w", encoding="utf-8") as f:
            processed_entries = [e.dict() for e in entries_by_id.values() if e.business.fsq_place_id in seen_ids]
            json.dump(processed_entries, f, indent=2, ensure_ascii=False)
        print(f"💾 Progress saved to {temp_file} (up to entry {i})")

    # Save final combined file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in entries_by_id.values()], f, indent=2, ensure_ascii=False)

    print(len(seen_ids), "entries already processed.")
    print(f"\n🎯 All evaluations saved to {output_file}")


def run_gemma_evaluation(input_file: str, output_file: str, temp_file: str = "temp_results.json") -> None:
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        raw_data = raw_data["results"]

        # Map all entries by fsq_place_id
    entries_by_id = {entry["business"]["fsq_place_id"]: GemmaEntry(**entry) for entry in raw_data}

    print(f"entries to process: {entries_by_id}")

    seen_ids: Set[str] = set()

    # Resume from checkpoint if temp file exists
    if os.path.exists(temp_file):
        with open(temp_file, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        # Update entries_by_id with already processed results
        for entry_data in temp_data:
            e = GemmaEntry(**entry_data)
            entries_by_id[e.business.fsq_place_id] = e
            seen_ids.add(e.business.fsq_place_id)

        print(f"🔄 Resuming from checkpoint: {len(seen_ids)} entries already processed")

    # Process remaining entries
    for i, entry in enumerate(entries_by_id.values(), start=1):
        if entry.business.fsq_place_id in seen_ids:
            print(f"⏩ Skipping entry {i} (already processed): {entry.business.name}")
            continue

        print(f"\n🔎 Evaluating entry {i}: {entry.business.name}")

        if not entry.business.normalized_description or not entry.parsed_suggestions:
            print("  ⚠️ Skipped (missing description or suggestions)")
        else:
            evaluation = evaluate_domains(entry.business.normalized_description, entry.parsed_suggestions)
            if evaluation:
                entry.evaluation = evaluation
                print(f"  ✅ Done: {len(evaluation.evaluations)} domains evaluated")
            else:
                print("  ❌ No evaluation returned")

        seen_ids.add(entry.business.fsq_place_id)

        # Update checkpoint (save only processed entries)
        with open(temp_file, "w", encoding="utf-8") as f:
            processed_entries = [e.dict() for e in entries_by_id.values() if e.business.fsq_place_id in seen_ids]
            json.dump(processed_entries, f, indent=2, ensure_ascii=False)
        print(f"💾 Progress saved to {temp_file} (up to entry {i})")

    # Save final combined file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in entries_by_id.values()], f, indent=2, ensure_ascii=False)

    print(len(seen_ids), "entries already processed.")
    print(f"\n🎯 All evaluations saved to {output_file}")