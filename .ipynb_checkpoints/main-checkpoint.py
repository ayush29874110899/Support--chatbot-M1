import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

##Loading my own llama-3b finetuned model.
@textbase.chatbot("talking-bot")
def on_message(message_history, state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")

    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, "PrinceAyush/Support-chatbot-llama7b")

    PROMPT = """Below is the instruction.As a Mental Health Support Chatbot, your primary goal is to provide guidance and support to individuals seeking mental health advice. Your responses should be empathetic, non-judgmental, and focused on promoting emotional well-being.
    ### Instruction:
    {}
    ### Response:""".format(message_history)

    inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cuda()

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
        bot_response=tokenizer.decode(s)

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history,
        model="gpt-3.5-turbo",
    )

    return bot_response, state
