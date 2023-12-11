import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERT(nn.Module):
    def __init__(self, bert, num_classes, dropout):
        super(BERT, self).__init__()
        self.bert = bert
        # Define additional layers for custom classification head
        self.additional_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Weight initialization (XavierGlorot initialization)
        for layer in self.additional_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.additional_layers(outputs[1])
        return logits

# Specify the classes mapping
classes_dict = {0: 'normal', 1: 'hatespeech', 2: 'offensive'}

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the model (handle DataParallel)
model_path = './model/classifier.pth'
loaded_model = torch.load(model_path)

# If the model was trained with DataParallel, unwrap it
if isinstance(loaded_model, nn.DataParallel):
    model_state_dict = loaded_model.module.state_dict()
else:
    model_state_dict = loaded_model.state_dict()

# Create an instance of the BERT class
model = BERT(bert=bert, num_classes=len(classes_dict), dropout=0.0)
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

while(True):
    # Get user input
    prompt = input('Please enter the sentence to be judged: ')

    # Tokenize and run inference
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    output_class = torch.argmax(outputs).item()

    # Print the result
    print(f"Predicted class: {classes_dict[output_class]}")
