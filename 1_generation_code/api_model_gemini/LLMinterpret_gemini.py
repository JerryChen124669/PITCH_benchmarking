import json
import os
import re
import time
from google import genai
from google.genai import types

# ==========================================
# 1. SETUP GEMINI API & DIRECTORIES
# ==========================================
# Load the key from the environment
API_KEY = os.environ["GEMINI_API_KEY"]  # set in your environment; never hard-code a key
client = genai.Client(api_key=API_KEY)

# Define your model target
student_model_id = "gemini-3.1-pro-preview"

DATA_DIR = "data"
RESULTS_DIR = "results"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTION TO PARSE DYNAMIC PROMPTS
# ==========================================
def parse_prompt_file(content):
    """
    Extracts the system prompt section dynamically and groups 
    the remaining text as the user prompt context.
    """
    # Regex designed to find text between '--- SYSTEM PROMPT ---' and the next markdown section header.
    # It gracefully ignores structural tags if present in your files.
    match = re.search(r'(?:\\s*)?--- SYSTEM PROMPT ---\s*\n(.*?)\n(?:\\s*)?--- ', content, re.DOTALL)
    
    if match:
        system_prompt = match.group(1).strip()
        end_of_sys_prompt = content.find(system_prompt) + len(system_prompt)
        user_content = content[end_of_sys_prompt:].strip()
        # Strip structural source tag artifacts from the start of user content if present
        user_content = re.sub(r'^(?:\\s*)', '', user_content)
    else:
        # Fallback split logic if regex fails to match completely
        if "--- SYSTEM PROMPT ---" in content:
            parts = content.split("--- SYSTEM PROMPT ---")[1].split("--- ")
            system_prompt = parts[0].strip()
            user_content = "--- " + "--- ".join(parts[1:])
        else:
            # Complete fallback default if formatting is totally missing
            system_prompt = "You are an expert computational biologist tasked with interpreting decision tree classification rules."
            user_content = content
            
    return system_prompt, user_content

# ==========================================
# 3. RUN THE MULTI-RUN AUTOMATED PIPELINE
# ==========================================
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: The directory '{DATA_DIR}' does not exist. Please create it and add your .txt prompts.")
    exit(1)

# Find all .txt files in the data directory
prompt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

if not prompt_files:
    print(f"⚠️ No .txt files found in the '{DATA_DIR}' folder.")
    exit(0)

print(f"Found {len(prompt_files)} prompt file(s) in '{DATA_DIR}'. starting execution pipeline...")

for file_idx, filename in enumerate(prompt_files, 1):
    file_path = os.path.join(DATA_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    output_file = os.path.join(RESULTS_DIR, f"{base_name}_results.jsonl")
    
    print(f"\n📂 [{file_idx}/{len(prompt_files)}] Processing file: {filename}")
    
    # Read raw prompt file content
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    # Split the content dynamically into system instruction and context
    system_prompt, user_content = parse_prompt_file(raw_content)
    
    # Process 5 runs per prompt file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for run in range(1, 6):
            print(f"  -> Run {run}/5 evaluating via {student_model_id}...")
            
            # Re-initialize the generation config dynamically with the isolated system prompt
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3 
            )
            
            try:
                # API Call using the modern SDK setup
                response = client.models.generate_content(
                    model=student_model_id,
                    contents=user_content,
                    config=generation_config
                )
                response_text = response.text

                usage = response.usage_metadata
                if usage:
                    print(f"     Tokens used: {usage.prompt_token_count} (Prompt) | {usage.candidates_token_count} (Generated)")
                        
            except Exception as e:
                print(f"  ❌ Error during Run {run}: {e}")
                response_text = f"Error processing request: {str(e)}"

            # Format entry payload matching your requirements
            final_output_row = {
                "id": base_name,
                "run": run,
                "final_answer": response_text
            }
            
            # Save line directly to the specific JSONL output file
            outfile.write(json.dumps(final_output_row) + "\n")
            outfile.flush() 
            
            # Delay to safely respect rate limits
            time.sleep(4) 
            
    print(f"🏁 Saved all 5 runs for '{filename}' to -> {output_file}")

print(f"\n🎉 Success! All generations are complete.")