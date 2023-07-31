# Import necessary libraries
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import gradio as gr
import torch

# Global Constants - A string containing the instruction prompt, which will be used in the chat_bot function
PROMPT = """Below is the instruction.As a Mental Health Support Chatbot, your primary goal is to provide guidance and support to individuals seeking mental health advice. Your responses should be empathetic, non-judgmental, and focused on promoting emotional well-being.
### Instruction:
   {}
### Response:"""

# Load Tokenizer and Model - Initialize the LlamaTokenizer and LlamaForCausalLM models
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = LlamaForCausalLM.from_pretrained(
    "openlm-research/open_llama_3b",
    load_in_8bit=True, # Load the model in 8-bit mode to reduce memory consumption, to run on cpu comment this line and change devicemap to 'cpu'.
    device_map="cuda", # Specify the device to be GPU (cuda) for inference
)

# Load PeftModel - Load the PeftModel for the provided LlamaForCausalLM model
model = PeftModel.from_pretrained(model, "PrinceAyush/Support-chatbot-llama7b")

# Function to generate chatbot response - Define the chat_bot function that takes text as input and generates a response using the model
def chat_bot(txt):
    prompt_text = PROMPT.format(str(txt)) # Replace the placeholder in PROMPT with the provided text
    inputs = tokenizer(prompt_text, return_tensors="pt") # Tokenize the prompt_text

    # Move the input_ids tensor to the appropriate device - Check if GPU (cuda) is available, and if so, move the input_ids tensor to the GPU using .cuda(). Otherwise, keep it on the CPU using .cpu().
    if torch.cuda.is_available() and model.device.type == "cuda":
        input_ids = inputs["input_ids"].cuda()
    else:
        input_ids = inputs["input_ids"].cpu()

    # Set generation configuration - Specify the generation parameters for the model
    generation_config = GenerationConfig(
        temperature=0.6, # Control the randomness of generated text (higher value means more random, lower value means more focused)
        top_p=0.95, # The threshold for nucleus sampling (higher value includes more diverse candidates)
        repetition_penalty=1.15, # Penalty for repeating tokens in generated text
    )

    print("Generating...") # Print a message to indicate the start of the generation process

    # Generate the response using the model - Call the .generate() method on the model with the input_ids and generation_config
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128, # Limit the number of tokens in the generated response
    )

    for s in generation_output.sequences: # Iterate through the generated sequences
        bot_response = tokenizer.decode(s) # Decode the generated sequence to get the final response

    return bot_response # Return the generated response

# Gradio Interface - Set up the Gradio interface to interact with the chat_bot function
out_text = gr.outputs.Textbox() # Define the output component as a Textbox
gr.Interface(chat_bot, 'textbox', out_text, title='Demo of finetuned LLaMA 7B Model').launch(share=True) # Launch the Gradio interface with the chat_bot function and the defined output component
