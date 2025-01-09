# -*- coding: utf-8 -*-
"""chiara copy of christina_training_model_CS224Nfinalproj_march12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17IKDpIBzRvA_DshS-_2nSWKHXZagsDGg
"""

!pip install transformers
!pip install datasets
!pip install huggingface_hub

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base", num_labels = 3)

from datasets import load_dataset, DatasetDict
tweets = load_dataset('chiarab/final-train', use_auth_token=True)

# Take random examples for train and validation
finetune_train = tweets['train'].shuffle(seed=1111).select(range(30))
#finetune_val = tweets['train'].shuffle(seed=1111).select(range(2219, 2695))
#finetune_test = tweets['train'].shuffle(seed=1111).select(range(2695, 3170))

#tweets['train'].to_csv('file1.csv' , index = True)
print(finetune_train)
#print(tweets)

from torch.utils.data import DataLoader

train_finetune_dataloader = DataLoader(finetune_train, batch_size=16)
#eval_finetune_dataloader = DataLoader(finetune_val, batch_size=16)

print(train_finetune_dataloader)

def get_target(val):
  print(val)
  vec = np.zeros((1,3))
  print(vec)
  vec[0][val] = 1
  return torch.from_numpy(vec)

import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

num_epochs = 3
num_training_steps = 3 * len(train_finetune_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in enumerate(train_finetune_dataloader):
      print(batch['Sentiment'])
      input = batch['OriginalTweet '][1]
      labels = get_target(batch['Sentiment'])
      inputs = tokenizer(input, return_tensors="pt")
      outputs = model(**inputs, labels=labels)
      outputs.loss.backward()
      optimizer.step()
      lr_scheduler.step()
      progress_bar.update(1)

    # validation
    loss = 0
    model.eval()
    for batch_i, batch in enumerate(eval_finetune_dataloader):
        with torch.no_grad():
          input = batch['OriginalTweet '][0]
          labels = get_target(batch['Sentiment'])
          inputs = tokenizer(input, return_tensors="pt")
          outputs = model(**inputs, labels=labels)
        loss += outputs.loss
    
    avg_val_loss = loss / len(finetune_val)
    print(f"Validation loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            },
            f"epoch_{epoch}.pt"
        )

#our trained model accuracy

correct = 0
wrong = 0

model.eval()
for tweet in finetune_test:
  inputs = tweet['id']
  tokenized_inputs = tokenizer(inputs, return_tensors="pt")
  outputs = model(**tokenized_inputs)
  labels = [0, 1, 2]
  prediction = torch.argmax(outputs.logits)
  if prediction == tweet['sentiment_category']:
    correct += 1
  else:
    wrong += 1

print("Finetuned Accuracy:")
print(correct / (correct + wrong))

correct = 0
wrong = 0

# Initialize the regular tokenizer
tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
# Initialize the regular model
model2 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base", num_labels = 3)

model2.eval()
for tweet in finetune_test:
  inputs = tweet['id']
  tokenized_inputs = tokenizer2(inputs, return_tensors="pt")
  outputs = model2(**tokenized_inputs)
  labels = [0, 1, 2]
  prediction = torch.argmax(outputs.logits)
  if prediction == tweet['sentiment_category']:
    correct += 1
  else:
    wrong += 1

print("Regular Accuracy:")
print(correct / (correct + wrong))
print(correct + wrong)