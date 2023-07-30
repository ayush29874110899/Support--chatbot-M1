from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import gradio as gr
import torch

# Global Constants
PROMPT = """Below is the instruction.As a Mental Health Support Chatbot, your primary goal is to provide guidance and support to individuals seeking mental health advice. Your responses should be empathetic, non-judgmental, and focused on promoting emotional well-being.
### Instruction:
   {}
### Response:"""


# Set the device to CPU explicitly
torch.device("cpu")

# Load Tokenizer and Model
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = LlamaForCausalLM.from_pretrained(
    "openlm-research/open_llama_3b",
    load_in_8bit=True,
    device_map="cpu", #Change to CPU or auto based on the system specification
)
model = PeftModel.from_pretrained(model, "PrinceAyush/Support-chatbot-llama7b")

# Function to generate chatbot response
def chat_bot(txt):
    prompt_text = PROMPT.format(str(txt))
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    # Check for GPU support
    if torch.cuda.is_available():
        input_ids = inputs["input_ids"].cuda()
    else:
        input_ids = inputs["input_ids"].cpu()

    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    print("Generating...")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
    )
    for s in generation_output.sequences:
        bot_response = tokenizer.decode(s)
    return bot_response

# Gradio Interface
out_text = gr.outputs.Textbox()
gr.Interface(chat_bot, 'textbox', out_text, title='Demo of finetuned LLaMA 7B Model').launch(share=True)
