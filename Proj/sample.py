import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AdamW, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Initialize Tokenizers and Models
bert_model_name = 'sanjayyy/newBert'
simcse_model_name = 'sanjayyy/newSimCSE'

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
simcse_model = AutoModel.from_pretrained(simcse_model_name)
hidden_size = bert_model.config.hidden_size

class CombinedModel(nn.Module):
    def __init__(self, bert_model, simcse_model, hidden_size, num_labels):
        super(CombinedModel, self).__init__()
        self.bert_model = bert_model
        self.simcse_model = simcse_model

        # Combining BERT and SimCSE embeddings
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

        # Single linear layer for classification
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT embeddings
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_cls = bert_outputs[0][:, 0]  # Use [CLS] token

        # SimCSE embeddings
        simcse_outputs = self.simcse_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        simcse_cls = simcse_outputs[0][:, 0]  # Use [CLS] token

        # Concatenate [CLS] tokens from both BERT and SimCSE
        combined_cls = torch.cat((bert_cls, simcse_cls), dim=-1)  # Concatenate BERT and SimCSE [CLS]

        # Project to hidden size
        combined_cls = self.projection(combined_cls)
        combined_cls = self.relu(combined_cls)
        combined_cls = self.dropout(combined_cls)

        # Final classification layer
        logits = self.classifier(combined_cls)
        return logits
    
    def get_cls_token(self, input_ids, attention_mask, token_type_ids):
        # Extract [CLS] token embeddings for contrastive loss
        with torch.no_grad():
            bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            bert_cls = bert_outputs[0][:, 0]  # [CLS] from BERT
            
            simcse_outputs = self.simcse_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            simcse_cls = simcse_outputs[0][:, 0]  # [CLS] from SimCSE

            combined_cls = torch.cat((bert_cls, simcse_cls), dim=-1)  # Concatenated [CLS]
        return combined_cls

# model = CombinedModel(bert_model,simcse_model,hidden_size,num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 3  # Adjust as needed
model = CombinedModel(bert_model, simcse_model, hidden_size, num_labels).to(device)
repo_id = "sanjayyy/combinedModel"
filename = "best_model_full.pth"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
checkpoint = torch.load(model_path, map_location='cpu')

# Check if the model is wrapped in DataParallel and extract the state_dict if necessary
if isinstance(checkpoint, torch.nn.parallel.DataParallel):
    checkpoint = checkpoint.module.state_dict()

# Load the state dict
model.load_state_dict(checkpoint)
model.eval() 
tokenizer = BertTokenizer.from_pretrained('/Users/sanjay/Downloads/bert')  # Adjust if using a different tokenizer

def predict_class(sentence):
    # Preprocess the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Forward pass on CPU
    with torch.no_grad():
        # Get the raw output logits directly from the model
        logits = model(inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids'))
        
        # Get the predicted class by taking the index of the max logit
        label_map = {0: 'Ham', 1: 'Spam', 2: 'Phishing'}  # Make sure these match your class labels
        predicted_class = label_map[torch.argmax(logits, dim=-1).item()]
    
    return predicted_class


# Example usage
sentence = '''

Hey Find me is this phishing mail or not:
50% offer festival season, call this number to get instant offer


'''
predicted_class = predict_class(sentence)
print(f"Predicted class: {predicted_class}")




# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# prompt = prompt_template.format(sentence=sentence, predicted_class=predicted_class)