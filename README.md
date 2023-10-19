# Sentiment Analysis with DistilBERT

## Overview

This project implements sentiment analysis using the DistilBERT model, leveraging PyTorch and the Hugging Face Transformers library. The model is fine-tuned on a dataset comprising 10,000 IMDB movie reviews.

## Project Components

1. **Data Preprocessing:**
   - The dataset is preprocessed using the DistilBERT tokenizer to convert raw text into a format suitable for model training.

2. **Model Architecture:**
   - The DistilBERT model is employed for sentiment analysis. The architecture, including hyperparameters, is defined to suit the task.

3. **Training Process:**
   - The model is trained using a TrainingArguments object, incorporating gradient accumulation and a learning rate schedule for effective training.

4. **Evaluation Metrics:**
   - The model's performance is assessed on a separate test dataset, with metrics such as accuracy and F1-score computed to gauge its effectiveness.

5. **Model Sharing:**
   - The trained model is shared and made accessible via the Hugging Face Hub, facilitating easy integration into other projects.

6. **Inference with Pipeline API:**
   - The Pipeline API is utilized for sentiment analysis on new data using the uploaded model. The model demonstrates accurate classification of text polarity as positive or negative.

## Evaluation Metrics

- **Accuracy:** Measures the overall correctness of the model's predictions.
- **F1-Score:** A metric that balances precision and recall, providing a comprehensive evaluation of model performance.

## Usage

1. **Training:**
   - Fine-tune the DistilBERT model on your dataset by following the training process outlined in the code.

2. **Evaluation:**
   - Assess the model's performance on a test dataset using the provided evaluation metrics.

3. **Model Sharing:**
   - Share your trained model on the Hugging Face Hub for easy accessibility and integration.

4. **Inference:**
   - Use the Pipeline API to perform sentiment analysis on new text data, benefiting from the accurate predictions of the trained model.

## Dependencies

- PyTorch
- Hugging Face Transformers
- Other dependencies as specified in the project code.

## Acknowledgments

- The project leverages the power of the DistilBERT model and the Hugging Face Transformers library.
- Inspiration and guidance from the natural language processing, machine learning, and deep learning communities.

## Streamlit Link - 

https://huggingface.co/Satyajithchary/Sentiment_Analysis_using_BERT_With_10000_Samples/tree/main/
