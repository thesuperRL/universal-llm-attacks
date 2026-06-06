from fastchat.conversation import get_conv_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/Users/ryanli/Documents/GitHub/Paper Recreations/universal-llm-attacks/models/vicuna-7b-v1.5"

# Token ids for "ASSISTANT:"
ASSISTANT_ENCODING = [319, 1799, 9047, 13566, 29901]


def find_tokenlist_location(input_ids, tokenlist):
    ptr = 0
    start = -1
    for i in range(len(input_ids)):
        if ptr == len(tokenlist):
            return start, i
        if input_ids[i].item() == tokenlist[ptr]:
            ptr += 1
            if ptr == 1:
                start = i
        else:
            ptr = 0
            start = -1
    raise ValueError("Token list not found in tokenized prompt")

def compute_loss_single(input_ids, labels, model, grad = True):
    device = next(model.parameters()).device
    if not grad:
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                labels=labels.to(device),
            )
    else:
        out = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
        )
    return out.loss.item()

def compute_loss_avg(query_targets, model, grad = True):
    losses = []
    for item in query_targets:
        losses.append(compute_loss_single(item["input_ids"], item["labels"], model, grad))
    return sum(losses) / len(losses), losses

def compute_loss_with_suffix_grad(item, model):
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    input_ids = item["input_ids"].to(device)
    labels = item["labels"].to(device)
    suffix_start = item["suffix_start_idx"]
    suffix_end = item["suffix_end_idx"]

    # full sequence embeddings, detached except suffix slice
    embeds = embed_layer(input_ids).detach().clone()
    suffix_embeds = embeds[:, suffix_start:suffix_end, :].detach().requires_grad_(True)
    embeds[:, suffix_start:suffix_end, :] = suffix_embeds

    attention_mask = torch.ones_like(input_ids, device=device)

    out = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
    )

    out.loss.backward()

    # gradient with regards to each suffix slot
    suffix_grad = suffix_embeds.grad.detach().clone()

    model.zero_grad(set_to_none=True)

    return out.loss.item(), suffix_grad.squeeze(0)