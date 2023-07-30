# Mental Health Support Chatbot Project

## Introduction

The Mental Health Support Chatbot project aims to develop an empathetic and non-judgmental chatbot that provides guidance and support to individuals seeking mental health advice. The chatbot utilizes a language model trained on a diverse dataset containing conversations related to mental health from various sources.

## Data Collection and Preprocessing

The dataset used for training the chatbot was collected from mental health forums, support groups, and online communities. It comprises 6,365 rows of conversations between users and the chatbot assistant. The data was loaded from a CSV file and underwent several preprocessing steps:

1. **Removing Unnecessary Columns**: Columns such as `questionID`, `questionTitle`, `topic`, and `therapistInfo` were removed as they were not relevant for the chatbot's development.

2. **Extracting Human and Assistant Text**: The conversation text was split into parts spoken by the human user and the chatbot assistant, allowing for better organization of data.

3. **Cleaning Text**: Basic text cleaning was performed to eliminate unwanted characters and symbols.

4. **Splitting Data into Questions and Answers**: The dataset was divided into input questions and corresponding chatbot-generated answers to form input-output pairs for model training.

## Model Architecture and Training

The chatbot utilizes the `LlamaForCausalLM` model from the OpenLM research repository. The model is pretrained on a large corpus of text data and further optimized for int8 training using Progressive Encoder-Freezing Training (PEFT). The PEFT components help enhance the model's performance during training.

The model training involves tokenization of the input data and generating instructions, inputs, and responses for each data point. The `transformers.Trainer` class is used to handle the training process, including batch processing, data collation, and logging.

## Result and Deployment

After training the chatbot on the mental health dataset, the final model is saved in the "Mentalhealth_chatbot" directory. The trained model is ready for deployment to provide mental health support and guidance to users seeking assistance with their emotional well-being.

## Acknowledgements

We would like to extend our heartfelt thanks to **CofactoryAI** for providing the frontend tool **textbase**. Their easy-to-use chatbot UI greatly simplified the development process, allowing us to focus on building the chatbot's functionalities without worrying about the UI. We truly appreciate their contribution to the project.

## Conclusion

The Mental Health Support Chatbot project successfully developed a chatbot capable of empathetically responding to users' mental health concerns. Although data preparation and model training could be further refined, the project achieved its primary goal of building a supportive and non-judgmental resource for individuals seeking mental health advice.

The trained chatbot can be integrated into various platforms, such as websites and mobile applications, to offer valuable support and guidance to users on their journey to emotional well-being. The project contributes to the advancement of AI-driven mental health support and showcases the potential of language models in delivering empathetic and helpful responses to sensitive topics.
