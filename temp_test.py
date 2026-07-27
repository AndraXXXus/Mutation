import sqlite3
import time
import json
import random
import itertools
from multiprocessing import Process, cpu_count
import argparse
import csv
import os
import sys
import re
from pathlib import Path
import pandas as pd
from openai import OpenAI
import spot

# Initialize directories
timestamp = int(time.time())
current_tempOut = Path("tempOut/" + str(timestamp))
current_tempOut.mkdir(parents=True, exist_ok=True)
current_tempOut_str = str(current_tempOut) + "/"

# Global Prompts
INSTRUCTION = """
Return EXACTLY one LTL formula as a single line.
The output must be directly parseable as an LTL formula using the Spot API.
"""

BASIC = """
You are a Linear Temporal Logic ( LTL ) Parser. Your task is to convert a given natural language statement to an LTL formula, using the provided mapping of natural language phrases to atomic propositions.

LTL Symbols :
    - AND : &
    - OR : |
    - NOT : !
    - IMPLIES : ->
    - BIIMPLICATION : <->
    - NEXT : X
    - EVENTUALLY : F
    - ALWAYS : G
    - UNTIL : U

Natural Language statement : {requirement}
Atomic Propositions : {atomic_proposition}
"""

ARTEMIS = """
You are an expert in translating natural language to linear temporal logic (LTL). Your job is to translate natural language to LTL. You must only use LTL operators and atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).
Recall that in LTL, G = globally, F = eventually, V = releases, X = next, U = until. You may use boolean operators (e.g., !, &, |, ->, <->) and can only use atomic propositions (NO NUMERICAL COMPARISON OPERATORS ALLOWED).

Inputs consist of:
1. unstructured natural language (string)
2. atomic proposition dictionary
    
The Outputs consist of:
1. output_LTL
    
Provide a list of the top 1 most likely translations (ordered by most likely first to least likely last) in the above output format for the following:

    'input_natural_language':{requirement},
    'atomic_propositions':{atomic_proposition}
"""

ADARULE = """
Task:
Translate natural language (NL) sentences into Linear Temporal Logic (LTL) formulas accurately.
Your answers always need to follow the output format.

Rules:
The converted formula should only contain atomic statements and operators.
Use standard LTL syntax and operators: G (globally), F (eventually), X (next), U (until), R (release), ! (negation), & (conjunction), | (disjunction), -> (implication), <-> (equivalence).
G means "globally": G a indicates that a is true in all future states.
F means "finally": F a indicates that a will eventually be true in some future state.
X means "next": X a is true if a is true in the next state.
U means "until": a U b is true if a remains true until b becomes true.
R means "release": a R b means b must be true until the moment when a is true and b is true. Once a is true, b can no longer be true. If a is never true, then b must always remain true.
Remember especially that the brackets match, we stipulate that each atomic formula is followed by a space. 
Do not change atomic propositions in NL.

Guidelines:
When translate "never", use  G!;
When translate "every time", "always", "all the time", use  G;
When translate "at certain moment", "eventually", "in the futhure" , "sooner or later", use  F;
When expressing "both A and B" or A and B will happen together at some moment, use A ∧ B;
When translate  "A or B holds"  or at some moment at least one of  A , B will be true , use A | B;
When the sentence is a discription of the system state, it means the state is always satisfied, so use G;
When the sentence is a discription of a and b happens, then finally,c and d will happen, you should considered the situation that the state "a and b" never satisfied , (e.g. G(!(a & b)) );
When translate "it is going to happen that a after b", use b -> F (a);
When translate  "Never (a) after (b)",use G((b) -> G(!(a)));
When translate  "Whenever (a), then (b)", use G((b) -> (a));
When translate  "A never happens", use G(!A);
For "After A, B happens"， Weak interpretation (B may happen even if A doesn’t)， use G(A -> F(B)),  Strong interpretation (A must happen first, then B), use !A | F(A & F(B));
For sentences like "Whenever A or B, then eventually C or D", use G((A | B) -> F(C | D));
For sentences like "At some point (A), and later (B)", use F(A & F(B));
For "First A and B, then eventually C or D" , use !(A & B) | F((A & B) & F(C | D));
For "After A and B, eventually C", use G((A & B) -> F(C));

Natural Language: {requirement}
Atomic Propositions: {atomic_proposition}

Please response in plain text format. DO NOT use markdown, latex or any other formats.
Please response in the following format, and replace the '[LTL formula]' with the LTL formula translated from the natural language sentences:
So the final LTL translation is: [LTL formula].FINISH
"""

model = "Qwen/Qwen3.5-27B"
DB_PATH = "experiment_results.db"
STALL_TIMEOUT_SECONDS = 600  

def normalize_formula(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""

def ask_chatgpt(client: OpenAI, model: str, prompt: str, requirement: str, atomic_proposition: str, temperature: float) -> str:
    content = globals()[prompt].format(
        requirement=requirement,
        atomic_proposition=atomic_proposition,
    ) + INSTRUCTION
    
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        do_sample=True,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return normalize_formula(response.choices[0].message.content or "")

def semantically_equivalent(formula_a: str, formula_b: str):
    try:
        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)
        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()
    except Exception as exc:
        msg = str(exc)
        if "syntax error" in msg.lower():
            with open(current_tempOut_str + "output_log.txt", "a", encoding="utf-8") as f:
                print(f"Syntax error; excluding from accuracy:\n  Error: {exc}", file=f)
            return None
        with open(current_tempOut_str + "output_log.txt", "a", encoding="utf-8") as f:
            print(f"Warning: could not compare formulas:\n  Error: {exc}", file=f)
        return None


def run_medium_duration_script(task):
    """Processes all 20 retakes in memory for a single parameter combination."""
    ind = task['experiment_index']
    requirement = task['original_NL']
    ground_truth = task['Spot_LTL']
    atomic_proposition = task['APs']
    prompt = task['prompt_type']
    temp = task['temperature']

    output = []
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

    for retake in range(20):
        time.sleep(random.uniform(0.5, 1.5))  # Simulate processing time
        model_response = ask_chatgpt(client, model, prompt, requirement, atomic_proposition, temp)
        og_model_response = model_response
        
        if prompt == "ADARULE":
            model_response = model_response.replace("So the final LTL translation is: ", "")
            model_response = model_response.replace(".FINISH", "").strip()

        equivalent = semantically_equivalent(ground_truth, model_response)
        equivalent2 = "None" if equivalent is None else equivalent

        output.append({       
            'PROMPTTYPE': prompt,
            "experiment_index": ind,
            "Requirement": requirement,
            "Ground Truth": ground_truth,
            "Atomic Proposition": atomic_proposition,
            "Original Response": og_model_response,
            "Response": model_response,
            "Equivalent": equivalent2
        })

    return output

data_points = []
    # Make sure this path matches your directory setup
with open("Batch9/final_df_downsampled.csv", mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file, delimiter=';')
    for row in csv_reader:
        row['experiment_index'] = int(csv_reader.line_num)  # Ensure the index is an integer
        data_points.append(row)

prompt_types = ["BASIC", "ARTEMIS", "ADARULE"]
temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_data = data_points[111]

print(test_data)
for prompt in prompt_types:
    for temp in temperatures:
        task = {}

        task['experiment_index'] = test_data['experiment_index']
        task['original_NL'] = test_data['original NL']
        task['Spot_LTL'] = test_data['Spot LTL']
        task['APs'] = test_data['APs']
        task['prompt_type'] = prompt
        task['temperature'] = temp

        out = run_medium_duration_script(task)

        print(prompt,temp,[x['Response'] for x in out])  
        