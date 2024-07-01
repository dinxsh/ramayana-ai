from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Sample context and question
context = "The Ramayana is an ancient Indian epic composed by the sage Valmiki."
question = "Who composed the Ramayana?"

# Encode inputs
inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Get model outputs
outputs = model(input_ids, attention_mask=attention_mask)
start_scores, end_scores = outputs.start_logits, outputs.end_logits

# Get the most likely start and end of the answer
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores) + 1

# Decode the answer
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
print(f"Answer: {answer}")
