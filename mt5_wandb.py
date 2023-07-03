from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, get_linear_schedule_with_warmup, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from datasets import load_dataset, load_metric, list_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import wandb

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm.auto import tqdm
import pandas as pd
import argparse
import time
import pickle
import numpy as np
import re
import utility as util


# commandline arguments
""" If you are running this in a jupyter notebook then you need to change the parser lines below
to equality e.g. traindata = "english_dataset/english_dataset.tsv" and then remove args. where applicable
"""
parser = argparse.ArgumentParser(description='Bipol Multilingual')
# --datatype: hasoc, trol, sema, semb, hos, olid (--olidtask = a, b or c)
# USAGE EXAMPLE in the terminal: python ht5.py --datatype trol
parser.add_argument('--task_pref', type=str, default="classification: ", help='Task prefix')
parser.add_argument('--savet', type=str, default='mt5.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='mt5.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--msave', type=str, default='mt5_saved_model', help='folder to save the finetuned model')
#parser.add_argument('--ofile1', type=str, default='outfile_', help='output file')
#parser.add_argument('--ofile2', type=str, default='outfile_', help='output file')
parser.add_argument('--lang', type=str, default='swedish', help='Language of the MAB dataset')
#parser.add_argument('--seed', type=int, default=1111, help='random seed')
#parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate') # bestloss at 0.0002; dpred- 1; weighted F1: 0.9386351943374753, micro F1: 0.9376623376623376; test #weighted F1: 0.8210865645981863, micro F1: 0.8227946916471507
#parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit')
#parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size') # smaller batch size for big model to fit GPU
args = parser.parse_args()

sweep_config = {
    'method': 'bayes', #'random', grid
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'learning_rate': {
            'max': 0.001, 'min': 0.00002 #'values': [ 5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [4, 8]
        },
        'epochs':{
            "max": 20, "min": 10 #'values':[1, 2, 3]
        }
    }
}

sweep_defaults = {
    'learning_rate': 0.00002,
    'batch_size': 8,
    'epochs':10
}

sweep_id = wandb.sweep(sweep_config, project="LREC Bias")

#mab_swedish = '/home/shared_data/bipol/mab_swedish/'

#def f1_score_func(preds, labels):
 #   preds_flat = []
  #  preds_flat_ = ['1' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
   # preds_flat.extend(preds_flat_)
    #labels_flat = labels            # only for consistency
    #return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted"), f1_score(labels_flat, preds_flat, average="micro")

def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    #preds_flat.extend(preds_flat_)
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="macro")

# def accuracy_score_func(preds, labels):
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return accuracy_score(labels_flat, preds_flat, normalize='False')

def confusion_matrix_func(preds, labels):
    preds_flat = []
    preds_flat_ = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    # print(confusion_matrix(labels, preds_flat))  # cell(r,c) 0,0- TN, 1,0- FN, 1,1- TP, 0,1- FP
    tn, fp, fn, tp = confusion_matrix(labels, preds_flat).ravel()
    return tn, fp, fn, tp


    print(confusion_matrix(labels, preds_flat))


def train():    # train_data, train_tags
    """One epoch of a training loop"""
    ### Take care of data
    wandb.init(config=sweep_defaults)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #wandb.init()

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small") # instead of T5Tokenizer
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate) #, betas=(0.7, 0.99))
    
    #traindata, devdata, testdata = util.get_data(args.datatype, args.datayear, augment_traindata=False)
    # Comment out the below if preprocessing not needed
    #traindata = util.preprocess_pandas(traindata, list(traindata.columns))
    #valdata = util.preprocess_pandas(devdata, list(devdata.columns))
    train_df = pd.read_csv(f"/home/shared_data/bipol/new/{args.lang}_mab_train.csv", header=0)
    train_df = util.preprocess_pandas(train_df, list(train_df.columns))
    train_df['comment_text'] = args.task_pref + train_df['comment_text'].astype(str)
    eval_df = pd.read_csv(f"/home/shared_data/bipol/new/{args.lang}_mab_val.csv", header=0)
    eval_df = util.preprocess_pandas(eval_df, list(eval_df.columns))
    eval_df['comment_text'] = args.task_pref + eval_df['comment_text'].astype(str)
    train_data = train_df['comment_text'].values.tolist()
    val_data = eval_df['comment_text'].values.tolist()

    ### Label column
    label_dict = {}         # For associating raw labels with indices/nos
    possible_labels = train_df.label.unique()
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)       # NOT: 0; HOF: 1
    train_df['label'] = train_df.label.replace(label_dict)                 # replace labels with their nos
    train_df['label'] = train_df['label'].apply(str)  # string conversion
    eval_df['label'] = eval_df.label.replace(label_dict)                 # replace labels with their nos
    eval_df['label'] = eval_df['label'].apply(str)  # string conversion
    print("Trainset distribution: \n", train_df['label'].value_counts())                           # check data distribution

    train_tags = train_df['label'].values.tolist()
    val_tags = eval_df['label'].values.tolist()

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_data)*wandb.config.epochs)

    args.msave = args.lang + args.msave             # prepend language to model folder name
    best_loss = None
    best_model = None
    args.epochs = wandb.config.epochs
    for epoch in range(1, args.epochs + 1):
        print("Training... epoch {}/{}".format(epoch, args.epochs))
        epoch_start_time = time.time()
        epoch_loss, train_steps, train_loss = 0, 0, 0

        # tokenizer.encode() converts the text to a list of unique integers before returning tensors
        einput_ids = tokenizer(train_data, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = einput_ids.input_ids, einput_ids.attention_mask
        labels = tokenizer(train_tags, padding=True, truncation=True, return_tensors='pt').input_ids

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        train_tensor = TensorDataset(input_ids, attention_mask, labels)
        train_sampler = RandomSampler(train_tensor)
        args.batch_size = wandb.config.batch_size
        train_dataloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=args.batch_size)

        model.train()  # Turn on training mode
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            batch_input_ids, batch_att_mask, batch_labels = batch
            loss = model(input_ids=batch_input_ids, attention_mask=batch_att_mask, labels=batch_labels).loss
            wandb.log({'train_batch_loss':loss.item()})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_steps += 1
        epoch_loss = train_loss / train_steps
        wandb.log({'avg_train_loss':epoch_loss})

        # Evaluate
        val_loss, predictions, true_vals = evaluate(val_data, val_tags, model, tokenizer, device)
        val_f1, val_f1_w, val_f1_mac = f1_score_func(predictions, true_vals)
        wandb.log({'val_fscore':val_f1_mac,'avg_val_loss':val_loss})
        tn, fp, fn, tp = confusion_matrix_func(predictions, true_vals)       # useful for calculating the error rate
        epoch_time_elapsed = time.time() - epoch_start_time
        print("tn, fp, fn, tp: ", tn, fp, fn, tp)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Elapsed time: {:.4f} '.format(epoch, epoch_loss, val_loss, epoch_time_elapsed) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}') # metric_sc['f1']))    , elapsed time: {epoch_time_elapsed}
        with open('outfile_mt5small.txt', "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Elapsed time: {:.4f} '.format(epoch, epoch_loss, val_loss, epoch_time_elapsed) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}' + "\n" + f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}' + "\n")
        #if not best_val_wf1 or val_f1_w > best_val_wf1:
        if not best_loss or val_loss < best_loss:
                #with open(args.savet, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                #No need to save the models for now so that they don't use up space 
                #torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
            #best_model = model
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.msave)  # transformers save
            tokenizer.save_pretrained(args.msave)
            best_loss = val_loss

    # return epoch_loss


def evaluate(val_data, val_tags, model, tokenizer, device):
    """One epoch of an evaluation loop"""
    print("Evaluation...")
    epoch_loss, val_steps, val_loss = 0, 0, 0

    # tokenizer.encode() converts the text to a list of unique integers before returning tensors
    einput_ids = tokenizer(val_data, padding=True, return_tensors='pt')
    input_ids, attention_mask = einput_ids.input_ids, einput_ids.attention_mask
    labels = tokenizer(val_tags, padding=True, return_tensors='pt').input_ids

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    val_tensor = TensorDataset(input_ids, attention_mask, labels) #, ids)
    val_sampler = SequentialSampler(val_tensor)
    args.batch_size = wandb.config.batch_size
    val_dataloader = DataLoader(val_tensor, sampler=val_sampler, batch_size=args.batch_size)
    predictions = []

    model.eval()  # Turn on evaluation mode
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            batch_input_ids, batch_att_mask, batch_labels = batch
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_mask, labels=batch_labels)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
            for a in prediction:            # pick each element - no list comprehension
                predictions.append(a)

            val_loss += outputs.loss.item()
            val_steps += 1
    true_vals = val_tags
    epoch_loss = val_loss / val_steps
    return epoch_loss, predictions, true_vals


#if __name__ == '__main__':

with open('outfile_mt5small.txt', "a+") as f:
    s = f.write(f"\n\n {args.lang} \n")

        #train_loss = train(train_data, train_tags)
        # val_loss, predictions, true_vals = evaluate(val_data, val_tags) # val_ids added for Hasoc submission
wandb.agent(sweep_id, count=2, function=train)

