from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Create a sentence with a masked word
text = "[CLS] I want to [MASK] a new language. [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = tokenized_text.index("[MASK]")
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Find the predicted token and replace the mask
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Replace the mask with the predicted token
tokenized_text[masked_index] = predicted_token

# Join the tokens back to a string
generated_sentence = ' '.join(tokenized_text).replace(' [SEP]', '.').replace('[CLS] ', '')

print(generated_sentence)
