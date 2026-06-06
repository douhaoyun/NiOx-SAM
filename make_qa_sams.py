"""
It uses the DeepSeek API for generating Q&A pairs for NiOx modification molecules in perovskite research.
@author: Yutang Li
Modified for NiOx modification molecules study
"""
import multiprocessing
import os
import re
import json
import random
import time
import tqdm
import glob
import datetime
import copy
import multiprocessing
from functools import partial
from openai import OpenAI, APIError  # DeepSeek is compatible with OpenAI SDK

# Constants
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = ""  # Replace with your actual DeepSeek API key
MODEL_DEEPSEEK = "deepseek-chat"  # Or use the specific model name provided by DeepSeek

# Path configurations
cur_dirname = os.path.dirname(__file__)
PROCESS = 16  # Reduced parallel processes for stability

# NiOx-specific prompts
NIOX_QUESTION_PROMPT = """You are an expert in perovskite solar cells research, specifically focusing on NiOx hole transport layer modification with molecules.

Given the following research context about NiOx layer modification in perovskite solar cells, generate 3 diverse and meaningful questions that can be answered based on this context.

Context:
{CONTEXT}

Requirements:
1. Focus specifically on molecular modification of NiOx layers
2. Questions should cover different aspects: molecular structure, modification methods, performance effects, characterization techniques
3. Make questions specific and technical
4. Avoid generic questions that could apply to any perovskite research

Return your response in JSON format:
{{
    "questions": [
        {{"id": 1, "text": "question text 1"}},
        {{"id": 2, "text": "question text 2"}},
        {{"id": 3, "text": "question text 3"}}
    ]
}}"""

NIOX_ANSWER_PROMPT = """You are an expert in perovskite solar cells research. Answer the following question specifically about NiOx layer molecular modification based on the provided research context.

Question: {QUESTION}

Research Context about NiOx modification:
{CONTEXT}

Additional Supplementary Information about general perovskite and NiOx properties:
{SUP_CONTENT}

Requirements:
1. Provide a comprehensive and technical answer
2. Focus specifically on molecular aspects of NiOx modification
3. Cite specific details from the research context when possible
4. If the context doesn't contain enough information, provide general knowledge but indicate this
5. Keep the answer focused and avoid unnecessary details

Answer:"""

SELECT_QUESTION_PROMPT = """You are evaluating questions for a Q&A system about NiOx layer molecular modification in perovskite solar cells.

Given the following research context and candidate questions, select the best question that:
1. Is most specifically relevant to NiOx molecular modification
2. Can be well-answered by the provided context
3. Has the highest technical depth and research value
4. Is not too broad or generic

Research Context:
{CONTEXT}

Candidate Questions:
{QUESTIONS}

Rate each question from 1-10 based on the criteria above and return in JSON format:
{{
    "questions": [
        {{"id": 1, "score": 8, "reason": "brief explanation"}},
        {{"id": 2, "score": 6, "reason": "brief explanation"}},
        {{"id": 3, "score": 9, "reason": "brief explanation"}}
    ]
}}"""

def extract_and_parse_json(response):
    """Extract and parse JSON from a response."""
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    json_str = json_match.group(1) if json_match else response.strip()
    json_str = re.sub(r'(\$[^\$]*\$)', lambda m: m.group(1).replace('\\', '\\\\'), json_str)
    json_str = json_str.replace('\\"', '"').replace("\\'", "'")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return 'errformat'

def contains_niox_content(content):
    """Check if content contains relevant NiOx modification information."""
    niox_keywords = [
        'niox', 'niox', 'NiOx', 'NIOX', 'nickel oxide', 
        'hole transport', 'HTL', 'perovskite', 'modification',
        'molecular', 'SAM', 'self-assembled', 'interface'
    ]
    
    if isinstance(content, dict):
        content_str = str(content).lower()
    else:
        content_str = content.lower()
    
    # Check if content contains sufficient NiOx-related information
    keyword_count = sum(1 for keyword in niox_keywords if keyword in content_str)
    return keyword_count >= 2  # Require at least 2 relevant keywords

def generate_niox_question(context, model_name):
    """Generate questions about NiOx modification using DeepSeek."""
    try:
        # Skip if context doesn't contain relevant NiOx information
        if not contains_niox_content(context):
            return 'skip'
            
        instruction = NIOX_QUESTION_PROMPT.replace("{CONTEXT}", str(context))
        
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY, 
            base_url=DEEPSEEK_BASE_URL
        )

        completion = client.chat.completions.create(
            model=model_name,
            stream=False,
            messages=[
                {"role": "system", "content": "You are a perovskite research expert specializing in NiOx modification."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content
        json_response = extract_and_parse_json(response)
        
        if json_response == "errformat":
            return 'errformat'
        return json_response['questions']

    except APIError as api_error:
        print(f"generate_niox_question API error: {api_error}")
        time.sleep(10)  # Reduced sleep time for DeepSeek
        return 'apierror'
    except Exception as e:
        print(f"generate_niox_question Unexpected error: {e}")
        return 'unexpectederror'

def generate_niox_answer(question, context, sup_content, model_name):
    """Generate answers about NiOx modification using DeepSeek."""
    try:
        instruction = NIOX_ANSWER_PROMPT.replace("{QUESTION}", question)\
                                        .replace("{CONTEXT}", str(context))\
                                        .replace("{SUP_CONTENT}", sup_content)
        
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY, 
            base_url=DEEPSEEK_BASE_URL
        )

        completion = client.chat.completions.create(
            model=model_name,
            stream=False,
            messages=[
                {"role": "system", "content": "You are a perovskite research expert."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content
        return response

    except APIError as api_error:
        print(f"generate_niox_answer API error: {api_error}")
        time.sleep(10)
        return 'apierror'
    except Exception as e:
        print(f"generate_niox_answer Unexpected error: {e}")
        return 'unexpectederror'

def select_best_question(question_list, context, model_name):
    """Select the best question for NiOx modification research."""
    try:
        instruction = SELECT_QUESTION_PROMPT.replace("{CONTEXT}", str(context))\
                                           .replace("{QUESTIONS}", json.dumps(question_list))
        
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY, 
            base_url=DEEPSEEK_BASE_URL
        )

        completion = client.chat.completions.create(
            model=model_name,
            stream=False,
            messages=[
                {"role": "system", "content": "You are a perovskite research expert."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content
        json_response = extract_and_parse_json(response)
        
        return json_response['questions']

    except APIError as api_error:
        print(f"select_best_question API error: {api_error}")
        time.sleep(10)
        return 'apierror'
    except Exception as e:
        print(f"select_best_question Unexpected error: {e}")
        return 'unexpectederror'

def generate_niox_qa(file_content, sup_content, model_name):
    """Generate Q&A pairs for NiOx modification research."""
    # Skip if content doesn't contain relevant NiOx information
    if not contains_niox_content(file_content):
        return None, None, 'skip'
    
    # 1. Generate candidate questions
    question = generate_niox_question(file_content, model_name)
    if question == 'skip':
        return None, None, 'skip'
        
    retry = 0
    while (question == 'errformat' or question == 'apierror' or question == 'unexpectederror') and retry < 2:
        question = generate_niox_question(file_content, model_name)
        retry += 1
    
    if question in ['errformat', 'apierror', 'unexpectederror', 'skip']:
        return None, None, question
    
    # 2. Select best question
    score = select_best_question(question, file_content, model_name)
    retry = 0
    while (score == 'errformat' or score == 'apierror' or score == 'unexpectederror') and retry < 2:
        score = select_best_question(question, file_content, model_name)
        retry += 1
    
    if score in ['errformat', 'apierror', 'unexpectederror']:
        return None, None, score
    
    try:
        score = sorted(score, key=lambda x: x['score'], reverse=True)
        q_idx = score[0]['id'] - 1
        
        # 3. Generate answer
        answer = generate_niox_answer(question[q_idx]['text'], file_content, sup_content, model_name)
        retry = 0
        while (answer == 'errformat' or answer == 'apierror' or answer == 'unexpectederror') and retry < 2:
            answer = generate_niox_answer(question[q_idx]['text'], file_content, sup_content, model_name)
            retry += 1
        
        if answer in ['errformat', 'apierror', 'unexpectederror']:
            return None, None, answer
            
        return question[q_idx]['text'], answer, 'success'
        
    except Exception as e:
        print(f"Error in question selection: {e}")
        return None, None, 'error'

def process_file(input_path, task_id):
    """Process a single file for NiOx modification Q&A generation."""
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            file_content = json.load(file)
        
        # Load supplementary content
        sup_content_path = os.path.join(cur_dirname, "supplementary", "sup_content.md")
        if os.path.exists(sup_content_path):
            with open(sup_content_path, "r", encoding='utf-8') as file:
                sup_content = file.read()
        else:
            sup_content = "General information about perovskite solar cells and NiOx hole transport layers."
        
        model = MODEL_DEEPSEEK
        
        # Generate Q&A
        question, answer, status = generate_niox_qa(file_content, sup_content, model)
        
        if status == 'skip':
            print(f"Skipped {input_path} - insufficient NiOx content")
            return
        elif status != 'success':
            print(f"Failed to generate Q&A for {input_path}: {status}")
            return
        
        # Prepare output data
        data = {
            "source_file": os.path.basename(input_path),
            "niox_question": question,
            "niox_answer": answer,
            "generation_date": datetime.datetime.now().isoformat(),
            "model_used": model
        }
        
        # Determine output directory based on task
        if task_id == "task-1":
            output_dir = os.path.join(cur_dirname, "task-1", "task1-niox-qa")
        else:  # task-2
            output_dir = os.path.join(cur_dirname, "task-2", "task2-niox-qa")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed {input_path}")
            
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

if __name__ == "__main__":
    # Configure paths based on your setup
    base_path = r"E:\a\dou\SAM\LLM"
    
    task1_dir = os.path.join(base_path, "task-1")
    task2_dir = os.path.join(base_path, "task-2")
    
    # Task 1 input paths
    task1_input_dir = os.path.join(task1_dir, "task1-paper-info")
    task1_input_paths = []
    if os.path.exists(task1_input_dir):
        task1_input_paths = [os.path.join(task1_input_dir, f) for f in os.listdir(task1_input_dir) 
                           if f.endswith('.json')]
    
    # Task 2 input paths  
    task2_input_dir = os.path.join(task2_dir, "task2-paper-info")
    task2_input_paths = []
    if os.path.exists(task2_input_dir):
        task2_input_paths = [os.path.join(task2_input_dir, f) for f in os.listdir(task2_input_dir) 
                           if f.endswith('.json')]
    
    print(f"Found {len(task1_input_paths)} Task 1 files")
    print(f"Found {len(task2_input_paths)} Task 2 files")
    
    # Process Task 1 files
    if task1_input_paths:
        print("Processing Task 1 files...")
        process_file_with_params = partial(process_file, task_id="task-1")
        
        with multiprocessing.Pool(processes=min(PROCESS, len(task1_input_paths))) as pool:
            list(tqdm.tqdm(
                pool.imap_unordered(process_file_with_params, task1_input_paths), 
                total=len(task1_input_paths)
            ))
    
    # Process Task 2 files
    if task2_input_paths:
        print("Processing Task 2 files...")
        process_file_with_params = partial(process_file, task_id="task-2")
        
        with multiprocessing.Pool(processes=min(PROCESS, len(task2_input_paths))) as pool:
            list(tqdm.tqdm(
                pool.imap_unordered(process_file_with_params, task2_input_paths), 
                total=len(task2_input_paths)
            ))
    
    print("NiOx modification Q&A generation completed!")