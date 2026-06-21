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
    model_path, dtype=dtype, local_files_only=True
).to(device)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=dtype, local_files_only=True
).to(device)

for p in ref_model.parameters():
    p.requires_grad_(False)

policy_model.train()
ref_model.eval()


def compute_response_mask(start_pos, end_pos, seq_len):
    seq_pt = torch.arange(seq_len, device=device).unsqueeze(0)
    return (seq_pt >= start_pos.unsqueeze(1)) & (seq_pt < end_pos.unsqueeze(1))


def compute_logp(logits, input_ids, response_mask):
    logp = F.log_softmax(logits, dim=-1)

    shift_logp = logp[:, :-1, :]
    labels = input_ids[:, 1:]

    token_logp = shift_logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    resp_logp = (token_logp * response_mask[:, 1:]).sum(-1)
    return resp_logp


def compute_loss(pi_chose_logp, pi_rej_logp, ref_chose_logp, ref_rej_logp):
    logp = (pi_chose_logp - pi_rej_logp) - (ref_chose_logp - ref_rej_logp)
    return -F.logsigmoid(logp * beta).mean()


small_dataset = load_small_data()

for i in range(0, len(small_dataset) - 1, batch_size):
    if i == 8:
        break

    chos_list = []
    rej_list = []
    start_pos = []
    chos_end_pos = []
    rej_end_pos = []
    for j in range(batch_size):
        item = small_dataset[i + j]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        p_ids = tokenizer(prompt, add_special_tokens=True, padding=False)["input_ids"]
        c_ids = tokenizer(chosen, add_special_tokens=False, padding=False)["input_ids"]
        r_ids = tokenizer(rejected, add_special_tokens=False, padding=False)[
            "input_ids"
        ]

        pc_ids = p_ids + c_ids
        pr_ids = p_ids + r_ids

        chos_list.append(pc_ids)
        rej_list.append(pr_ids)
        start_pos.append(len(p_ids))
        chos_end_pos.append(len(pc_ids))
        rej_end_pos.append(len(pr_ids))

    input_list = chos_list + rej_list
    input_batch = tokenizer.pad(
        [{"input_ids": ids} for ids in input_list],
        padding=True,
        return_tensors="pt",
    ).to(device)

    pi_outputs = policy_model(
        input_ids=input_batch["input_ids"], attention_mask=input_batch["attention_mask"]
    )
    seq_len = pi_outputs.logits.shape[1]

    with torch.no_grad():
        ref_outputs = ref_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
        )
    # mask
    start_pos_pt = torch.tensor(start_pos).to(device)
    chose_resp_mask = compute_response_mask(
        start_pos_pt, torch.tensor(chos_end_pos).to(device), seq_len
    )

    rej_resp_mask = compute_response_mask(
        start_pos_pt, torch.tensor(rej_end_pos).to(device), seq_len
    )

    # policy
    policy_chose_logp = compute_logp(
        pi_outputs.logits[0:batch_size, :, :],
        input_batch["input_ids"][0:batch_size, :],
        chose_resp_mask,
    )

    policy_rej_logp = compute_logp(
        pi_outputs.logits[batch_size:, :, :],
        input_batch["input_ids"][batch_size:, :],
        rej_resp_mask,
    )

    # ref
    ref_chose_logp = compute_logp(
        ref_outputs.logits[0:batch_size, :, :],
        input_batch["input_ids"][0:batch_size, :],
        chose_resp_mask,
    )

    ref_rej_logp = compute_logp(
        ref_outputs.logits[batch_size:, :, :],
        input_batch["input_ids"][batch_size:, :],
        rej_resp_mask,
    )

    loss = compute_loss(
        policy_chose_logp, policy_rej_logp, ref_chose_logp, ref_rej_logp
    )
    print(loss.item())
    # todo: check begin and tail padding and whether correct when shift