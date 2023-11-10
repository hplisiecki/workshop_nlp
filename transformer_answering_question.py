from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load a fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Define the question and context
question = "Why did the chicken cross the road?"
context = "This is a joke."

# Encode the question and context
inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

# Get the input IDs and attention mask
input_ids = inputs["input_ids"].tolist()[0]
attention_mask = inputs["attention_mask"]

# Make the model predict the answer
with torch.no_grad():
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

# Find the tokens with the highest `start` and `end` scores
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1

# Convert the tokens to the answer string
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

print("Question:", question)
print("Answer:", answer)
