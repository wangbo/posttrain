import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from download_data import load_small_data

device = "cuda"
dtype = torch.bfloat16
model_path = "../model/Qwen2.5-0.5B-Instruct/"
beta = 0.1
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

policy_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    local_files_only=True
).to(device)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    local_files_only=True
).to(device)

for p in ref_model.parameters():
    p.requires_grad(False)

policy_model.train()
ref_model.eval()

def compute_logp(logits, input_ids, response_mask):
    logp = F.log_softmax(logits, dim=-1)

    shift_logp = logp[:, :-1,:]
    labels = input_ids[:, 1:]

    token_logp = shift_logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    resp_logp = (token_logp * response_mask[:,1:]).sum(-1)
    return resp_logp

def compute_loss(
        pi_chose_logp, 
        pi_rej_logp, 
        ref_chose_logp,
        ref_rej_logp):
    logp = (pi_chose_logp - pi_rej_logp) - (ref_chose_logp - ref_rej_logp)
    return - F.logsigmoid(logp * beta).mean()

small_dataset = load_small_data()

for i in range(len(small_dataset)):
    if i == 1000:
        break
    
    choose_list = []
    reject_list = []
    resp_idxs = []
    for j in range(batch_size):
        item = small_dataset[i + j]
        prompt = item['prompt']
        choose = item['choose']
        reject = item['reject']
        
        p_ids = tokenizer(prompt)
        c_ids = tokenizer(choose)
        r_ids = tokenizer(reject)
        choose_ids = p_ids + c_ids
        reject_ids = p_ids + r_ids

        choose_list.append(choose_ids)
        reject_list.append(reject_ids)
        resp_idxs.append(len(p_ids) - 1)

    choose_input_ids = torch.tesnor(choose_list, dtype=dtype).to(device)
    reject_input_ids = torch.tesnor(reject_list, dtype=dtype).to(device)
    resp_idxs = torch.tensor(resp_idxs, dtype=torch.long).to(device)
    
    pi_choose_logits = policy_model(input_ids = choose_input_ids)
    pi_reject_logits = policy_model(input_ids = reject_input_ids)

    with torch.no_grad():
        ref_choose_logits = ref_model(input_ids = choose_input_ids)
        ref_reject_logits = ref_model(input_ids = reject_input_ids)