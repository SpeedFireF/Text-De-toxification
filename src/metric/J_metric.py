from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification

import os
import gc
import tqdm
import torch
import argparse
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from tqdm.auto import trange


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_preds(preds, batch_size=1):
    print('Calculating style of predictions')
    results = []

    model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    for i in tqdm.tqdm(range(0, len(preds), batch_size)):
        batch = tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
        with torch.inference_mode():
            logits = model(**batch).logits
            result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        results.extend([1 - item for item in result])
    return results


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Function to calculate cosine similarity between two lists of sentence embeddings
def calculate_similarity(input_sentences, pred_sentences, model, tokenizer):
    print("Calculate the semantic similarity")
    # Tokenize input sentences
    input_encoded = tokenizer(input_sentences, padding=True, truncation=True, return_tensors='pt')
    pred_encoded = tokenizer(pred_sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings for input and prediction sentences
    with torch.no_grad():
        input_model_output = model(**input_encoded)
        pred_model_output = model(**pred_encoded)

    # Perform pooling for input and prediction sentences
    input_embeddings = mean_pooling(input_model_output, input_encoded['attention_mask'])
    pred_embeddings = mean_pooling(pred_model_output, pred_encoded['attention_mask'])

    # Normalize embeddings
    input_embeddings = F.normalize(input_embeddings, p=2, dim=1)
    pred_embeddings = F.normalize(pred_embeddings, p=2, dim=1)

    # Calculate cosine similarity between the corresponding pairs of embeddings
    similarity_scores = []
    for i in range(len(input_sentences)):
        similarity = 1 - cosine(input_embeddings[i], pred_embeddings[i])
        similarity_scores.append(similarity)

    return similarity_scores


def detokenize(x):
    return x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",")").replace("( ", "(")


def do_cola_eval_transformers(preds):
    print('Calculating CoLA acceptability stats')
    path = "cointegrated/roberta-large-cola-krishna2020"

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    results = []
    bs = 1
    for i in trange(0, len(preds), bs):
        batch = [detokenize(t) for t in preds[i: i + bs]]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = torch.softmax(model(**inputs).logits, -1)[:, 0].cpu().numpy()
            results.append(out)
    return np.concatenate(results)


def J(inputs, preds):
    from transformers import AutoModel
    accuracy_by_sent = classify_preds(preds)
    accuracy = sum(accuracy_by_sent)/len(preds)
    print(accuracy)
    cleanup()

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Calculate cosine similarity between input and prediction sentences
    similarity_scores = calculate_similarity(inputs, preds, model, tokenizer)

    similarity = sum(similarity_scores) / len(similarity_scores)
    print(similarity)
    cleanup()

    fluency = sum(do_cola_eval_transformers(inputs)) / len(inputs)
    print(fluency)
    cleanup()

    # count metrics
    joint = accuracy * similarity * fluency

    print('| ACC | SIM | FL | J |\n')
    print(f'|{accuracy:.4f}|{similarity:.4f}|{fluency:.4f}|{joint:.4f}|\n')
    return joint
