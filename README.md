# Text-Classification
This repository contains is implementation of text classification use case.

# README

## Finetuning Roberta-base for Email Spam Classification using LoRA

This repository contains code for finetuning the Roberta-base model on a text classification task using LoRA (Low-resource Adaptation) with a dataset comprising emails (subject + message) and labeled as spam or ham.

### Task
- **Text Classification**: The task involves classifying emails into spam or ham categories.

### Model
- **Roberta-base**: Pretrained Roberta-base model is utilized for the task.

### Dataset
- **Dataset**: The dataset used is [likhith231/enron_spam_small](https://huggingface.co/datasets/likhith231/enron_spam_small). It consists of two primary columns: 'Text' and 'Label', containing 1000 samples for training and 1000 samples for testing, making it suitable for binary text classification tasks.

### Libraries Used
- `transformers`: For utilizing and fine-tuning the Roberta-base model.
- `huggingface-hub`: For accessing the model and tokenizer from the Hugging Face model hub.
- `peft`: For training and evaluation of the model.
- `datasets`: For handling and processing the dataset.
- `evaluate`: For evaluating the model performance.
- `numpy`: For numerical computations.
- `torch`: For building and training neural networks.

### Training Details
- **Pretrained Model**: Roberta-base.
- **Total Parameters**: 125,313,028
- **Trainable Parameters**: 665,858
- **Trainable Parameter Percentage**: 0.531355766137899

### Hyperparameters
- **Weight Decay**: 0.01
- **Learning Rate**: 1e-3
- **Batch Size**: 4
- **Number of Epochs**: 10

### Results

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | No log        | 0.172788        | 0.957    |
| 2     | 0.194500      | 0.202991        | 0.956    |
| 3     | 0.194500      | 0.229950        | 0.958    |
| 4     | 0.038400      | 0.267390        | 0.954    |
| 5     | 0.038400      | 0.283116        | 0.963    |
| 6     | 0.007000      | 0.254960        | 0.961    |
| 7     | 0.007000      | 0.299375        | 0.961    |
| 8     | 0.007900      | 0.276321        | 0.966    |
| 9     | 0.007900      | 0.275304        | 0.967    |
| 10    | 0.002000      | 0.271234        | 0.967    |

### Usage
- Clone the repository.
- Install the required libraries listed in `requirements.txt`.
- Run the training script with appropriate configurations.

### Acknowledgments
- The dataset used in this project is provided by likhith231 on Hugging Face datasets hub.