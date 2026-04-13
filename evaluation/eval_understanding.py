import argparse
import json
import logging
import os
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Constants
MAX_TOKENS = 8192
OPENAI_CLIENT = None

# Global model client
MODEL_CLIENT = None
MODEL_CONFIG = None
TOKENIZER = None

NON_JUDGEMENT_JUDGE_PROMPT = """# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer's behavior across the entire interview in terms of non-judgement, i.e. whether the interviewer refrains from evaluating, validating, or passing verdicts on the interviewee's choices, experiences, or answers.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# What counts as judgement (bad)
- Positive evaluations: "that's impressive", "great", "good call", "makes sense", "smart approach", "you did the right thing"
- Negative evaluations: "that seems like a mistake", "that's unusual", "that's surprising"
- Implied verdicts: framing that assumes one answer is better than another, or that the user's choice was unexpected or notable in a way that implies a standard being applied

# What counts as non-judgement (good)
- Neutral reflections of content: "sounds like that took up most of the week", "it seems like that decision came up quickly"
- Reflections of experience without verdict: "that sounds like it was a stressful stretch", "seems like that was still unresolved"
- Asking follow-up questions without signalling whether the prior answer was right or wrong

# Evaluation Rubric

1: The interviewer frequently uses evaluative or validating language, making the interviewee's experience feel assessed rather than explored.
2: Evaluative language is common; the interviewer often signals approval or surprise in ways that imply a standard being applied.
3: Mixed — some turns are neutral but evaluative phrases appear regularly enough to be noticeable.
4: Mostly non-judgmental, with only occasional lapses (e.g., a stray "makes sense" or "good").
5: Consistently non-judgmental throughout; the interviewer never implies a verdict on the interviewee's choices or experiences.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final non-judgement score, citing specific examples if possible.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""

ACTIVE_UNDERSTANDING_JUDGE_PROMPT = """# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer's behavior across the entire interview in terms of active understanding, i.e. whether the interviewer's acknowledgments reflect the interviewee's actual experience (not just paraphrase facts), and whether the interviewer gives space to emotionally or experientially significant moments rather than pivoting away immediately.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# What counts as active understanding (good)
- Acknowledging the experience or feeling implied, not just the fact: "that sounds like it put you in a difficult position" rather than "so you switched tools"
- Slowing down when the interviewee shares something hard, uncertain, or emotionally loaded — asking about the experience before moving to the next topic
- Picking up on tone or subtext in the interviewee's answer and reflecting it back

# What counts as weak understanding (bad)
- Immediately pivoting to the next question after the interviewee shares something difficult or significant
- Acknowledgments that only paraphrase facts without reflecting the experience: "okay so you did X, now tell me about Y"
- Generic filler: "thanks for sharing", "got it", "interesting" — these show the interviewer heard the words but not the meaning
- Missing cues where the interviewee signals frustration, uncertainty, or effort

# Evaluation Rubric

1: Acknowledgments are absent or purely mechanical; the interviewer consistently pivots past emotionally significant moments without any reflection.
2: Acknowledgments mostly paraphrase facts; the interviewer rarely reflects the interviewee's experience or slows down for significant moments.
3: Mixed — some acknowledgments reflect experience, but the interviewer frequently misses cues or pivots too quickly.
4: Acknowledgments usually reflect the interviewee's experience; significant moments are mostly given space, with occasional lapses.
5: Acknowledgments consistently reflect the interviewee's actual experience; the interviewer reliably slows down and engages with emotionally or experientially significant moments before moving on.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final active understanding score, citing specific examples if possible.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""

NON_AFFIRMATION_JUDGE_PROMPT = """# Instruction
You will be given the full transcript of a semi-structured interview.

Evaluate the interviewer's behavior across the entire interview in terms of non-affirmation, i.e. whether the interviewer avoids sycophantic or affirming responses that praise, validate, or positively reinforce the interviewee's answers.
Do NOT score individual turns. The score should reflect a global, holistic judgment over the full interview.

# What counts as affirmation / sycophancy (bad)
- Explicit praise: "great answer", "that's really impressive", "wow", "excellent", "I love that"
- Validation of choices: "that makes total sense", "that was the right call", "smart approach"
- Positive reinforcement of sharing: "thanks so much for sharing that", "really appreciate your openness"
- Filler affirmations: "absolutely", "definitely", "of course", "exactly" used as openers before a question
- Excitement markers: "oh interesting!", "that's fascinating!" — signals that imply the answer was better or more interesting than others

# What counts as non-affirmation (good)
- Neutral acknowledgments that reflect content without evaluating it: "sounds like that shifted the timeline", "it seems like that added complexity"
- Moving to the next question without commenting on the quality of the prior answer
- Occasional neutral fillers ("okay", "right", "I see") used sparingly and not as substitutes for substantive acknowledgment

# Evaluation Rubric

1: Sycophantic or affirming language is pervasive — nearly every response praises or validates the interviewee.
2: Affirming language is frequent; the interviewer regularly compliments answers or validates choices in a way that feels like positive reinforcement.
3: Mixed — affirmations appear regularly but not constantly; some turns are neutral while others are clearly evaluative.
4: Mostly non-affirming, with occasional slips (e.g., a stray "great" or "absolutely") that don't dominate the tone.
5: Consistently non-affirming throughout; the interviewer never praises, validates, or positively reinforces answers.

# Response Format
{'type': 'object', 'properties': {'explanation': {'type': 'string', 'description': 'A brief reasoning behind the final non-affirmation score, citing specific examples if possible.'}, 'score': {'type': 'string', 'description': "The verdict label from the rubric: one of '1', '2', '3', '4', or '5'.", 'enum': ['1', '2', '3', '4', '5']}}, 'required': ['explanation', 'score']}
"""


def safe_parse_json(text: str):
    text = text.strip()
    if not text:
        return None

    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        parsed_dict = ast.literal_eval(text)
        if isinstance(parsed_dict, dict):
            return parsed_dict
    except Exception:
        pass

    return None


def initialize_model(config_path: Optional[str] = None):
    global MODEL_CLIENT, MODEL_CONFIG, TOKENIZER, OPENAI_CLIENT

    OPENAI_CLIENT = OpenAI()

    if config_path is None:
        MODEL_CLIENT = OPENAI_CLIENT
        MODEL_CONFIG = {
            "provider_name": "openai",
            "model_name": "gpt-4.1-nano",
            "generation_args": {
                "temperature": 0,
                "max_tokens": MAX_TOKENS
            }
        }
        logging.info("Using default OpenAI client")
        return

    with open(config_path, 'r') as f:
        MODEL_CONFIG = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "openai":
        MODEL_CLIENT = OPENAI_CLIENT
        logging.info(f"Using OpenAI client with model {MODEL_CONFIG.get('model_name')}")
    elif provider == "local":
        try:
            from vllm import LLM
            model_name = MODEL_CONFIG["model_name"]
            model_args = MODEL_CONFIG.get("model_args", {})
            logging.info(f"Initializing vLLM with model {model_name}")
            MODEL_CLIENT = LLM(model=model_name, **model_args)
            TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            logging.info("vLLM client and tokenizer initialized successfully")
        except ImportError:
            logging.error("vLLM not installed. Please install with: pip install vllm")
            raise
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_chat_log(user_id: str, base_path: str, mode: str) -> Optional[str]:
    if mode in ('sparkme', 'storysage'):
        chat_path = f"{base_path}/{user_id}/execution_logs/session_1/chat_history.log"
    elif mode in ('llmroleplay', 'freeform'):
        chat_path = f"{base_path}/{user_id}/interview_log.jsonl"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(chat_path):
        return None

    if mode in ('llmroleplay', 'freeform'):
        chat_log = []
        turn = 0
        with open(chat_path, 'r') as f:
            for obj in f.readlines():
                loaded_obj = json.loads(obj)
                if loaded_obj.get("user_message", ""):
                    chat_log.append(f"User: {loaded_obj.get('user_message')}")
                    turn += 1
                if loaded_obj.get("assistant_message", ""):
                    chat_log.append(f"Interviewer: {loaded_obj.get('assistant_message')}")
                if mode == 'freeform' and turn == 10:
                    break
        return "\n".join(chat_log)
    else:
        with open(chat_path, 'r') as f:
            return f.read()


def prepare_messages(chat_log: str) -> List[List[Dict]]:
    user_content = f"# Interview Transcript\n\n<interview_transcript>{chat_log}</interview_transcript>\n\n# Your Output\nRespond in JSON format with 'explanation' and 'score' fields.\n"
    return [
        [{"role": "system", "content": NON_JUDGEMENT_JUDGE_PROMPT},
         {"role": "user", "content": user_content}],
        [{"role": "system", "content": ACTIVE_UNDERSTANDING_JUDGE_PROMPT},
         {"role": "user", "content": user_content}],
        [{"role": "system", "content": NON_AFFIRMATION_JUDGE_PROMPT},
         {"role": "user", "content": user_content}],
    ]


METRIC_NAMES = ["non_judgement", "active_understanding", "non_affirmation"]


def main():
    parser = argparse.ArgumentParser(description='Evaluate interviewer understanding and non-judgement')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'])
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--sample-users-path', type=str,
                        default='analysis/sample_users_50.json')
    parser.add_argument('--output-path', type=str, default='output_understanding.json')
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--num-users', type=int, default=200)

    args = parser.parse_args()

    initialize_model(args.model_config)

    with open(args.sample_users_path, 'r') as f:
        sample_users = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    # Prepare all sessions
    sessions = []  # list of (user_id, messages_list)
    for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
        user_id = person_profile["User ID"]
        chat_log = load_chat_log(user_id, args.base_path, args.mode)
        if chat_log is None:
            logging.warning(f"No chat log found for user {user_id}, skipping")
            continue
        sessions.append((user_id, prepare_messages(chat_log)))

    all_results = []

    if provider == "openai":
        model_name = MODEL_CONFIG.get("model_name", "gpt-4.1-mini")
        gen_args = MODEL_CONFIG.get("generation_args", {"temperature": 0, "max_tokens": MAX_TOKENS})
        max_workers = MODEL_CONFIG.get("max_workers", 8)

        def call_openai(user_id: str, metric: str, messages: List[Dict]) -> Dict:
            for attempt in range(3):
                try:
                    response = MODEL_CLIENT.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        **gen_args,
                    )
                    text = response.choices[0].message.content.strip()
                    parsed = safe_parse_json(text)
                    if parsed:
                        return {"user_id": user_id, "metric": metric,
                                "score": parsed["score"], "explanation": parsed["explanation"], "error": False}
                    return {"user_id": user_id, "metric": metric, "score": 1, "explanation": text, "error": True}
                except Exception as e:
                    logging.warning(f"Attempt {attempt+1} failed for {user_id}/{metric}: {e}")
            return {"user_id": user_id, "metric": metric, "score": 1, "explanation": "max retries exceeded", "error": True}

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for user_id, messages_list in sessions:
                for metric, messages in zip(METRIC_NAMES, messages_list):
                    futures.append(executor.submit(call_openai, user_id, metric, messages))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            all_results.append(future.result())

    elif provider == "local":
        from vllm import SamplingParams

        all_messages = []
        all_user_ids = []
        for user_id, messages_list in sessions:
            all_messages.extend(messages_list)
            all_user_ids.append(user_id)

        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        for idx, output in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
            generated_text = output.outputs[0].text.strip()
            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1]
            parsed_response = safe_parse_json(generated_text)
            metric = METRIC_NAMES[idx % len(METRIC_NAMES)]
            if parsed_response:
                all_results.append({"user_id": all_user_ids[idx // len(METRIC_NAMES)], "metric": metric,
                                     "score": parsed_response["score"], "explanation": parsed_response["explanation"], "error": False})
            else:
                all_results.append({"user_id": all_user_ids[idx // len(METRIC_NAMES)], "metric": metric,
                                     "score": 1, "explanation": generated_text, "error": True})
    else:
        raise ValueError(f"Unknown provider: {provider}")

    with open(args.output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    from collections import defaultdict
    scores_by_metric = defaultdict(list)
    for r in all_results:
        if not r["error"]:
            scores_by_metric[r["metric"]].append(int(r["score"]))
    for metric, scores in scores_by_metric.items():
        avg = sum(scores) / len(scores) if scores else 0
        logging.info(f"{metric}: avg={avg:.2f} over {len(scores)} sessions")

    logging.info(f"Done. Results written to {args.output_path}")


if __name__ == "__main__":
    main()
