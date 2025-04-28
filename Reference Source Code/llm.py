from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to generate prompt
def construct_prompt_for_comments(nl, file_content):
    prompt = nl + f"\n\nHere are some functions:\n{file_content}\n\n"
    prompt += "Select the best function based on the query above:\n"
    return prompt

# Function to generate output using the model
def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"]

    input_ids = input_ids.to(model.device)
    attention_mask = inputs["attention_mask"]
    attention_mask = attention_mask.to(model.device)
    print(f"Tokens: {inputs}")
    print(f"Token Count: {input_ids.shape[1]}")

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=1024,  # Set a limit for the output length
        temperature=0.7,  # Control creativity
        top_p=0.9         # Sampling for diversity
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to parse and select the best function from the generated response
def select_best_function_from_response(response, file_content):
    # Split the content of file into functions
    functions = file_content.strip().split("\n\n")
    
    for func in functions:
        if func.strip() in response.strip():
            return f"The optimal function is:\n{func.strip()}"
    return "No optimal function found."

# Initialize model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Read content from output.txt
with open('query_responses_top5.txt', 'r') as file:
    file_content = file.read()

# Query you want to use for the generation
nl = "Select the optimal solution among the given solutions, add more details if needed"

# Construct the prompt for model generation
prompt = construct_prompt_for_comments(nl, file_content)

# Generate the model's response
response = generate_output(prompt)
print("Generated Response:\n", response)

# # Select the best function
# best_function = select_best_function_from_response(response, file_content)
# print(best_function)
