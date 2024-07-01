from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Read context from a text file with UTF-8 encoding
context_file_path = './../data/english/archive/ramayan.txt'
with open(context_file_path, 'r', encoding='utf-8') as file:
    context = file.read()

# Function to split the context into chunks
def chunk_context(context, max_length):
    tokens = tokenizer.tokenize(context)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# Chunk the context to fit within model's maximum input length
max_length = 512 - 50  # Allow some space for the question
context_chunks = chunk_context(context, max_length)

print("Type your question or type 'quit' to exit.")

while True:
    # Get question from the user
    question = input("Question: ")
    
    if question.lower() == 'quit':
        break

    answers = []
    for chunk in context_chunks:
        # Encode inputs
        inputs = tokenizer.encode_plus(question, chunk, return_tensors='pt')
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
        answers.append(answer)
    
    # Combine answers from different chunks (if needed)
    final_answer = ' '.join(answers)
    print(f"Answer: {final_answer}\n")
