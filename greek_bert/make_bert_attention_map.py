#!/usr/bin/env python3

import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model_dir = "/path/to/bert_output"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertModel.from_pretrained(model_dir, output_attentions=True, output_hidden_states=True)
model.eval()

# 1. Static token embedding neighbors
emb = model.embeddings.word_embeddings.weight.detach().cpu().numpy()

def nearest_neighbors(word, k=10):
    tok_id = tokenizer.convert_tokens_to_ids(word)
    sims = cosine_similarity([emb[tok_id]], emb)[0]
    topk = sims.argsort()[-k-1:][::-1]
    return [(tokenizer.convert_ids_to_tokens(i), float(sims[i])) for i in topk if i != tok_id]

print(nearest_neighbors("λόγος"))

# 2. Sentence embeddings
sentences = [
    "ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον",
    "πάντες ἄνθρωποι τοῦ εἰδέναι ὀρέγονται φύσει",
    "χαλεπὰ τὰ καλά",
]

vecs = []
for s in sentences:
    inputs = tokenizer(s, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    last_hidden = out.last_hidden_state[0]           # [seq, hidden]
    mask = inputs["attention_mask"][0].unsqueeze(-1) # [seq, 1]
    mean_pool = (last_hidden * mask).sum(0) / mask.sum()
    vecs.append(mean_pool.cpu().numpy())

vecs = np.vstack(vecs)

# 3. t-SNE projection
proj = TSNE(n_components=2, perplexity=min(5, len(vecs)-1), random_state=42).fit_transform(vecs)

plt.figure(figsize=(6,5))
plt.scatter(proj[:,0], proj[:,1])
for i, s in enumerate(sentences):
    plt.annotate(f"S{i+1}", (proj[i,0], proj[i,1]))
plt.title("Sentence Embeddings")
plt.show()

# 4. Attention heatmap
sentence = "πάντες ἄνθρωποι τοῦ εἰδέναι ὀρέγονται φύσει"
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

att = out.attentions[0][0, 0].cpu().numpy()  # layer 0, head 0
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(8,6))
plt.imshow(att, aspect="auto")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title("Attention Heatmap (Layer 1, Head 1)")
plt.colorbar()
plt.tight_layout()
plt.show()