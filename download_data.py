from datasets import load_dataset, Dataset

def download_small_data():
    N = 2000
    ds_stream = load_dataset(
        "Mxode/I_Wonder_Why-Chinese", "preference",
        split="train", streaming=True,
    )

    rows = list(ds_stream.take(N))
    small = Dataset.from_list(rows)

    def to_dpo(ex):
        rejected = ex["rejected-1"]
        if isinstance(rejected, list):
            rejected = rejected[0]
        return {
            "prompt":   ex["prompt"],
            "chosen":   ex["chosen"],
            "rejected": rejected,
        }

    small_dpo = small.map(to_dpo, remove_columns=small.column_names)
    small_dpo.to_json("./data/preference_small.jsonl", force_ascii=False)
    print(small_dpo[0])

def load_small_data():
    return load_dataset("json", data_files="./data/preference_small.jsonl", split="train")