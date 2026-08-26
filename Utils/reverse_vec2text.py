from transformers import CLIPTextModel, CLIPTokenizer
import torch
from statistics import mean
import argparse
import os

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

def nn_project(curr_embeds, embedding_layer, print_hits=False,top_k=5):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.token_embedding.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=top_k,
                                score_function=dot_score)
            
        top_k_resutlts = []
        for i in range(top_k):
            nn_indices = torch.tensor([hit[i]["corpus_id"] for hit in hits], device=curr_embeds.device)
            nn_indices = nn_indices.reshape((bsz,seq_len))
            top_k_resutlts.append(nn_indices)

    return top_k_resutlts

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Generate Images')
    parser.add_argument('--embedding_path', default='/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Results/Main/CLIP/text_text/len7/poisoned_model_2/Backdoor_Embedding_42.pt')   
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--len', default=1, type=int)
    parser.add_argument('--top_k', default=100, type=int)
    parser.add_argument('--backdoor_text_encoder', default="/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Models/Backdoor_Models/CLIP/text_text/len7/poisoned_model_2")
    parser.add_argument('--tokenizer', default='/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4/tokenizer')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:7')
    args = parser.parse_args()
    
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer)
    text_encoder = CLIPTextModel.from_pretrained(args.backdoor_text_encoder)
    embedding_layer = text_encoder.text_model.embeddings 
    
    cur_embedding = torch.load(args.embedding_path)
    cur_embedding = cur_embedding.unsqueeze(0)
    
    # forward projection
    nn_indices = nn_project(cur_embedding, embedding_layer, print_hits=True,top_k=args.top_k)
    
    for nn_indice in nn_indices:
        texts = tokenizer.batch_decode(nn_indice, skip_special_tokens=False)
        # encoded = tokenizer(
        #     texts,
        #     return_tensors="pt",
        #     add_special_tokens=True   # 默认会加上 <BOS> 和 <EOS>
        # )
        # if len(encoded.input_ids[0])!=9:
        #     continue
        print(texts)