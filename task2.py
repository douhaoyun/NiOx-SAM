from openai import OpenAI
from pathlib import Path
import os
import re
import json
import glob
import tqdm
from multiprocessing import Pool
from functools import partial

API_KEY = ""  
BASE_URL = "https://api.deepseek.com"  
MODEL_GPT = "deepseek-chat"  

INPUT_MD_DIR = r"E:\a\dou\SAM\LLM\mds"  
OUTPUT_CHUNKS_DIR = r"E:\a\dou\SAM\LLM\task-2\task2-chunks"  
OUTPUT_MOLECULES_DIR = r"E:\a\dou\SAM\LLM\task-2\task2-paper-info"  

def comfirm_json_string_gpt(json_string):

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        Fix the following string into a valid JSON format that can be parsed by json.loads.
        Output only the JSON string without any additional text.
        String: {json_string}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You are a JSON formatting expert."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error in comfirm_json_string_gpt: {e}")
        return json_string

def comfirm_json_string(response):

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    json_str = json_match.group(1) if json_match else response.strip()

    json_str = re.sub(r'[\\"\']', '', json_str)
    json_str = json_str.replace("\n", "").replace("\r", "")
    
    return json_str

def split_by_heading(markdown_text, heading_level='#'):

    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    split_texts = re.split(pattern, markdown_text)
    return [block.strip() for block in split_texts if block.strip() and len(block) > 50]  # 过滤过短块

def segment_classification(text_split):

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        Classify this text segment from a perovskite solar cell research paper into one of these categories:
        Abstract, Introduction, Materials and methods, Results and discussion, Conclusions, References.
        Output only the category name.
        Text: {text_split[:2000]}  
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You are an expert in classifying academic paper sections."},
                {"role": "user", "content": prompt}
            ],
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Classification API Error: {e}")
        return "Unknown"

def process_file(md_path, output_dir):

    try:
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as file:
            md_content = file.read()
        
        content_splits = []
        for level in ['#', '##', '###']:
            splits = split_by_heading(md_content, level)
            content_splits.extend(splits)
        
        chunks = []
        for idx, split_text in enumerate(content_splits):
            category = segment_classification(split_text)
            if ":" in category:
                category = category.split(":")[-1].strip()
            
            chunks.append({
                "id": idx + 1,
                "chunk": split_text,
                "category": category
            })
        
        output_path = os.path.join(output_dir, f"{Path(md_path).stem}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error processing {md_path}: {e}")

def get_niox_molecules(text):

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        You are analyzing a perovskite solar cell research paper. Extract ANY mention of molecules, functional groups, or materials used to modify or interface with the NiOx (nickel oxide) hole transport layer.
        
        Focus on:
        1. Specific molecule names (e.g., "2PACz", "MeO-2PACz", "PTAA", "PEDOT:PSS")
        2. Functional groups (e.g., "phosphonic acid", "carboxylic acid", "thiol")
        3. Materials used as interface layers with NiOx
        4. Brief description of their role or application
        
        Important: 
        - Extract even brief mentions (e.g., "we used PTAA on NiOx" is valuable)
        - If no NiOx-related modification is mentioned, return an empty JSON {{}}
        - Output format: {{"molecule/group": "description of application with NiOx"}}
        
        Text excerpt: {text[:3000]}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You extract molecular modification information for NiOx in perovskite solar cells."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Molecule extraction API Error: {e}")
        return "{}"

def extract_molecules_from_chunks(chunks_path):
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        molecules_dict = {}
        relevant_sections = 0
        
        for chunk in chunks:
            if chunk['category'] in ['Abstract', 'Introduction', 'Materials and methods', 'Results and discussion', 'Experimental']:
                result_text = get_niox_molecules(chunk['chunk'])
                cleaned_json = comfirm_json_string(result_text)
                
                try:
                    mol_data = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    fixed_json = comfirm_json_string_gpt(cleaned_json)
                    try:
                        mol_data = json.loads(fixed_json) if fixed_json else {}
                    except:
                        mol_data = {}
                
                if isinstance(mol_data, dict) and mol_data:
                    molecules_dict.update(mol_data)
                    relevant_sections += 1
        
        if not molecules_dict:
            molecules_dict = {"note": "No NiOx modification molecules found in this paper"}
        
        molecules_dict["_metadata"] = {
            "total_chunks_processed": len(chunks),
            "relevant_sections_found": relevant_sections,
            "molecules_extracted": len([k for k in molecules_dict.keys() if k != "_metadata" and k != "note"])
        }
        
        output_path = os.path.join(OUTPUT_MOLECULES_DIR, f"niox_{Path(chunks_path).stem}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(molecules_dict, f, ensure_ascii=False, indent=2)
            
        return len(molecules_dict) > 1  
    except Exception as e:
        print(f"Error processing {chunks_path}: {e}")
        return False

def chunk_done(json_dir):
    if not os.path.exists(json_dir):
        return set()
    jsons = os.listdir(json_dir)
    return set([json_name.replace('.json', '') for json_name in jsons])

def md_segment():

    os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)
    
    md_pattern = os.path.join(INPUT_MD_DIR, "**", "*.md")
    md_paths = glob.glob(md_pattern, recursive=True)
    print(f"Found {len(md_paths)} .md files in {INPUT_MD_DIR}")

    processed_files = chunk_done(OUTPUT_CHUNKS_DIR)
    md_paths = [p for p in md_paths if Path(p).stem not in processed_files]
    
    if not md_paths:
        print("All files already processed for segmentation.")
        return
    
    print(f"Processing {len(md_paths)} new files for segmentation...")
    
    with Pool(processes=min(4, len(md_paths))) as pool:
        process_func = partial(process_file, output_dir=OUTPUT_CHUNKS_DIR)
        list(tqdm.tqdm(
            pool.imap_unordered(process_func, md_paths),
            total=len(md_paths),
            desc="Segmenting MD files"
        ))

def extract_niox_molecules():
    os.makedirs(OUTPUT_MOLECULES_DIR, exist_ok=True)
    
    chunk_pattern = os.path.join(OUTPUT_CHUNKS_DIR, "*.json")
    chunk_paths = glob.glob(chunk_pattern)
    print(f"Found {len(chunk_paths)} chunk files in {OUTPUT_CHUNKS_DIR}")
    
    processed_files = chunk_done(OUTPUT_MOLECULES_DIR)
    chunk_paths = [p for p in chunk_paths if f"niox_{Path(p).stem}" not in processed_files]
    
    if not chunk_paths:
        print("All chunk files already processed for molecule extraction.")
        return
    
    print(f"Extracting molecules from {len(chunk_paths)} chunk files...")
    
    with Pool(processes=min(4, len(chunk_paths))) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(extract_molecules_from_chunks, chunk_paths),
            total=len(chunk_paths),
            desc="Extracting NiOx molecules"
        ))
    
    papers_with_info = sum(results)
    print(f"Extraction complete. Found NiOx modification info in {papers_with_info}/{len(chunk_paths)} papers.")

def main():
    print("="*60)
    print("NiOx Modification Molecule Extraction Pipeline")
    print("="*60)
    print(f"Input directory: {INPUT_MD_DIR}")
    print(f"Chunks output: {OUTPUT_CHUNKS_DIR}")
    print(f"Molecules output: {OUTPUT_MOLECULES_DIR}")
    print()
    
    print("STEP 1: Text Segmentation and Classification")
    print("-" * 40)
    md_segment()
    
    print("\nSTEP 2: NiOx-related Molecule Extraction")
    print("-" * 40)
    extract_niox_molecules()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()