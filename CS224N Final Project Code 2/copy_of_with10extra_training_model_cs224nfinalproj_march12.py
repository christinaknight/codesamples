# -*- coding: utf-8 -*-
"""Copy of with10extra_training_model_CS224Nfinalproj_march12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k69TZDudFD4EZxw2VZbNjtN6ImK-af9u
"""

!pip install transformers
!pip install datasets
!pip install huggingface_hub



from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels = 3)

from datasets import load_dataset, DatasetDict
tweets = load_dataset('chiarab/sorted-with-10-lessneu', use_auth_token=True)

print(len(tweets['train']))

# Take random examples for train and validation
finetune_train = tweets['train'].shuffle(seed=1111).select(range(840))
finetune_val = tweets['train'].shuffle(seed=1111).select(range(840, 1019))
finetune_test = tweets['train'].shuffle(seed=1111).select(range(1019, 1192))
#finetune_train = tweets['train'].shuffle(seed=1111).select(range(100))
#finetune_val = tweets['train'].shuffle(seed=1111).select(range(100, 130))
#finetune_test = tweets['train'].shuffle(seed=1111).select(range(130, 160))

print(finetune_train)

from torch.utils.data import DataLoader

print(finetune_train)
train_finetune_dataloader = DataLoader(finetune_train, batch_size=16)
eval_finetune_dataloader = DataLoader(finetune_val, batch_size=16)

print(train_finetune_dataloader)

def get_target(val):
  vec = np.zeros((1,3))
  vec[0][val] = 1
  return torch.from_numpy(vec)

import numpy as np
import scipy
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

num_epochs = 2
num_training_steps = 3 * len(train_finetune_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in enumerate(train_finetune_dataloader):
      inputs = batch['OriginalTweet'] # list of tweet texts
      labels = [batch['Sentiment'][i].item() for i in range(len(batch['Sentiment']))] # list of labels
      tokenized_inputs = [tokenizer(input, return_tensors="pt") for input in inputs] # list of tokenized tweets
      outputs = [model(**tokenized_inputs[i], labels=get_target(labels[i])) for i in range(len(tokenized_inputs))]
      [output.loss.backward() for output in outputs]
      optimizer.step()
      lr_scheduler.step()
      progress_bar.update(1)

    # validation
    loss = 0
    model.eval()
    for batch_i, batch in enumerate(eval_finetune_dataloader):
        with torch.no_grad():
          inputs = batch['OriginalTweet'] # list of tweet texts
          labels = [batch['Sentiment'][i].item() for i in range(len(batch['Sentiment']))] # list of labels
          tokenized_inputs = [tokenizer(input, return_tensors="pt") for input in inputs] # list of tokenized tweets

          outputs = [model(**tokenized_inputs[i], labels=get_target(labels[i])) for i in range(len(tokenized_inputs))]

        loss += sum([output.loss for output in outputs])
    
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
            f"lessneutral-epoch_{epoch}.pt"
        )

#our trained model accuracy

correct = 0
wrong = 0

model.eval()
for tweet in finetune_test:
  inputs = tweet['OriginalTweet']
  tokenized_inputs = tokenizer(inputs, return_tensors="pt")
  outputs = model(**tokenized_inputs)
  labels = [0, 1, 2]
  prediction = torch.argmax(outputs.logits)
  print(prediction)
  if prediction == tweet['Sentiment']:
    correct += 1
  else:
    wrong += 1

print("Finetuned Accuracy:")
print(correct / (correct + wrong))

correct = 0
wrong = 0

# Initialize the regular tokenizer
tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# Initialize the regular model
model2 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels = 3)

model2.eval()
for tweet in finetune_test:
  inputs = tweet['OriginalTweet']
  tokenized_inputs = tokenizer2(inputs, return_tensors="pt")
  outputs = model2(**tokenized_inputs)
  labels = [0, 1, 2]
  prediction = torch.argmax(outputs.logits)
  print(prediction)
  if prediction == tweet['Sentiment']:
    correct += 1
  else:
    wrong += 1

print("Regular Accuracy:")
print(correct / (correct + wrong))
print(correct + wrong)

tweets_dct = load_dataset('chiarab/dct-keyword-all', use_auth_token=True)
tweets_dct = tweets_dct['train']
tweets_vaccine = load_dataset('chiarab/vaccine-keyword-all', use_auth_token=True)
tweets_vaccine = tweets_vaccine['train']
print(tweets_dct)
print(finetune_test)

model.eval()
neg = 0
neu = 0
pos = 0

for tweet in tweets_vaccine:
  inputs = tweet['OriginalTweet']
  tokenized_inputs = tokenizer(inputs, return_tensors="pt")
  outputs = model(**tokenized_inputs)
  labels = [0, 1, 2]
  prediction = torch.argmax(outputs.logits)
  if prediction == 0:
    neg += 1
  elif prediction == 1:
    neu += 1
  else:
    pos += 1

print(neg)
print(neu)
print(pos)

print(prediction)