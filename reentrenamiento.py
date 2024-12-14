import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#data con párrafos y noticias completas
df = pd.read_excel("final_train.xlsx")

df['ETIQUETA'] = df["ETIQUETA"].apply(lambda x: 0 if x == "POSITIVO" else (1 if x == "NEGATIVO" else 2))

# modelo previamente fine-tuneado
model_path = 'v4modelo_finbert_reentrenado'
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only= True, num_labels = 3)

def tokenize_data(texts, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  
            max_length=max_length,
            padding='max_length',     
            truncation=True,          
            return_tensors='pt'       
        )
        input_ids.append(encoding['input_ids'].flatten())
        attention_masks.append(encoding['attention_mask'].flatten())

    return torch.stack(input_ids), torch.stack(attention_masks)

input_ids, attention_masks = tokenize_data(df['Body'].values, tokenizer)
labels = torch.tensor(df['ETIQUETA'].values, dtype=torch.long)

train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)


optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train() 
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

model.save_pretrained('v1final_model')
tokenizer.save_pretrained('v1final_model')