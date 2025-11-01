from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from losses import log_prob_loss, log_perplexity
from utils import get_forbidden_toks, filter_forbidden_toks 


def run_arca(args, model, tokenizer, embedding_table, output_str):
    top_k = args.arca_batch_size
    vocab_size, embedding_dim = embedding_table.shape

    forbidden_input_toks = get_forbidden_toks(
        args, 
        tokenizer,
        n_total_toks=vocab_size, 
        output=False, 
        output_str=output_str
    )
    output_toks = np.array(tokenizer(output_str)['input_ids'])
    args.output_length = output_toks.shape[0]

    curr_toks = np.random.choice(vocab_size, size=args.prompt_length + args.output_length, replace=True)
    curr_toks[args.prompt_length:] = output_toks
    curr_toks_tensor = torch.tensor(np.tile(curr_toks, (top_k, 1))).long().cuda()
    
    full_embeddings = torch.zeros(top_k, args.prompt_length + args.output_length, embedding_dim).cuda()
    for i in range(args.prompt_length + args.output_length):
        full_embeddings[:, i] = embedding_table[curr_toks[i]].unsqueeze(0).repeat(top_k, 1)

    labels = torch.cat([
        -100 * torch.ones(args.prompt_length).cuda().unsqueeze(0).repeat(top_k, 1), 
        curr_toks_tensor[:, args.prompt_length:]], dim=1).long()

    for _ in tqdm(range(args.arca_iters)):
        for tok_id in range(args.prompt_length):
            new_indices = np.random.choice(vocab_size, size=top_k, replace=True)
            full_embeddings[:, tok_id, :] = embedding_table[new_indices, :] 
            curr_toks_tensor[:, tok_id] = torch.tensor(new_indices).long().cuda()
            
            full_embeddings = full_embeddings.detach()
            if full_embeddings.requires_grad:
                full_embeddings.grad.zero_()
            full_embeddings.requires_grad = True
            full_embeddings.retain_grad()

            out = model(
                inputs_embeds=full_embeddings, 
                labels=labels
            )
            loss = log_prob_loss(out, labels, temp=1) + args.lam_perp * log_perplexity(out, curr_toks_tensor[:, :args.prompt_length])
            loss.backward(retain_graph=True)

            scores = torch.matmul(embedding_table, full_embeddings.grad[:, tok_id, :].mean(dim=0))
            best_scores_idxs = filter_forbidden_toks(scores.argsort(), forbidden_input_toks)
            full_embeddings= full_embeddings.detach()
            
            with torch.no_grad():
                full_embeddings[:, tok_id, :] = embedding_table[best_scores_idxs[:top_k], :]                
                curr_toks_tensor[:, tok_id] = best_scores_idxs[:top_k]
                
                out = model(inputs_embeds=full_embeddings)
                
                log_probs = F.log_softmax(out.logits[:, -1-args.output_length:-1, :], dim=2)
                batch_log_probs = torch.stack([
                    log_probs[i, torch.arange(args.output_length), curr_toks_tensor[i, args.prompt_length:]].sum() 
                    for i in range(top_k)
                ])
                batch_log_probs -= args.lam_perp * log_perplexity(out, curr_toks_tensor[:, :args.prompt_length], ret_all=True)
                
                best_batch_idx = batch_log_probs.argmax()
                best_idx = best_scores_idxs[best_batch_idx]
                
                curr_toks[tok_id] = best_idx.item()
                curr_toks_tensor[:, tok_id] = best_idx.item()
                full_embeddings[:, tok_id, :] = embedding_table[best_idx].unsqueeze(0).repeat(top_k, 1)
                
                gen_output = log_probs[best_batch_idx].argmax(dim=1)
                actual_output = curr_toks_tensor[0][args.prompt_length:]
                if (actual_output == gen_output).all().item():
                    curr_toks = curr_toks[:-args.output_length]
                    return curr_toks
    return None
