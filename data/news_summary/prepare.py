import os

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8
num_proc_load_dataset = num_proc
max_train_rows = 100
val_size = 0.01

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("argilla/news-summary", num_proc=num_proc_load_dataset)

    dataset["train"] = dataset["train"].select(range(min(max_train_rows, len(dataset["train"]))))

    split_dataset = dataset["train"].train_test_split(test_size=val_size, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset["train"].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for example in tqdm(dset, desc=f'writing {filename}'):
            ids = np.array(example['ids'], dtype=dtype)
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
