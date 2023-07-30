# Mental Health Support Chatbot Project

This project focuses on building a Mental Health Support Chatbot using state-of-the-art technologies, including Llama 3B language model, PEFT, LORA, and 8-bit model quantization. The chatbot aims to provide empathetic and non-judgmental responses to individuals seeking mental health advice, promoting emotional well-being and support. The project comprises Data Preparation, Model Training, and Quantization of the Model.

## Data Preparation

The data preparation phase involved collecting and preprocessing a dataset containing mental health conversations from various sources. The dataset consists of 6,365 rows of dialogues related to mental health. To ensure optimal training, the dataset was cleaned and formatted, removing noise, special characters, and irrelevant information.

To enhance the model's performance, data augmentation techniques were employed. Domain-specific language models were utilized to generate additional conversation examples, enabling the chatbot to respond effectively to a wider range of user queries.

## Model Training

For the model training, the Llama 3B language model was chosen due to its exceptional performance in natural language understanding. The model was fine-tuned on the prepared mental health dataset using hyperparameters such as batch size, learning rate, and gradient accumulation steps. The training process aimed to optimize the model's ability to generate appropriate and supportive responses based on user prompts.

## PEFT and LORA

In this project, PEFT (Parallel Efficient Transformers) and LORA (Locally Recurrent Adaptive Mechanism) techniques were incorporated to enhance the model's efficiency and performance. PEFT improves the model's scalability and training speed on multi-GPU systems. LORA, on the other hand, enhances the model's ability to capture long-range dependencies in the conversation context.

## Model Quantization

Due to resource constraints, the model was quantized in 8-bit format using model quantization techniques. Quantization reduces the model size and memory footprint, making it more feasible to deploy on devices with limited resources. The chatbot achieved satisfactory performance with the quantized model, allowing it to run efficiently on systems with lower RAM and GPU capacity.

## Model Training Environment

The model was trained on Google Colab, utilizing a virtual machine with 12GB CPU and 12GB T4 GPU RAM. Despite the resource limitations, the model training process yielded desirable results, demonstrating the effectiveness of the applied techniques in creating a functional and resource-efficient chatbot.

## Drawbacks of Model Quantization

While 8-bit model quantization provides significant benefits in terms of model size and resource consumption, it may result in a slight decrease in the model's precision and accuracy. The quantized model might not retain the exact same performance as the full-precision model. However, for the purposes of this project and the target application, the trade-off in performance is acceptable given the hardware constraints.

## How to Run the Application

To experience the Mental Health Support Chatbot application, follow these steps:

Step 1: Install the required dependencies by executing the following command in your terminal or command prompt:

```bash
pip install -r requirements.txt
```

Step 2: Execute the `runApp.py` script:

```bash
python runApp.py
```

Please note that the application requires a minimum system specification of 8 GB RAM and 6 GB of GPU to run efficiently.

## Test Prompts

Here are some example prompts that were tested on the Mental Health Support Chatbot:

1. "I've been feeling really anxious lately. What should I do to cope with it?"

2. "I'm feeling hopeless and don't see any point in living anymore."

3. "I can't sleep at night, and it's affecting my daily life."

4. "I'm having trouble concentrating, and I feel so overwhelmed."

5. "My friend told me they're feeling suicidal. What can I do to help them?"

## Conclusion

The Mental Health Support Chatbot project showcases the successful implementation of advanced technologies like PEFT, LORA, and 8-bit model quantization to build an efficient and supportive chatbot. While the model's quantization presents some trade-offs, it allows the chatbot to run effectively on devices with limited resources, making it accessible to a broader audience.

We encourage further exploration and improvement of the chatbot by leveraging larger and more diverse datasets and fine-tuning hyperparameters. Additionally, user feedback and continuous development will help enhance the chatbot's capabilities, providing better mental health support to users.

Finally, we express our gratitude to cofactoryai for their invaluable contribution by providing the frontend interface for the application, ensuring a user-friendly experience for the Mental Health Support Chatbot.

**Important Note: The chatbot is not a substitute for professional mental health advice or therapy. Users with severe mental health concerns should seek help from qualified professionals.**