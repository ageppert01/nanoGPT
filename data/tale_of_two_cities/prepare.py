import os
import requests

import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    data_url = 'https://www.gutenberg.org/cache/epub/98/pg98.txt'
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

    if not os.path.exists(input_file_path):
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    start = data.find('*** START OF THE PROJECT GUTENBERG EBOOK A TALE OF TWO CITIES ***')
    if start != -1:
        data = data[start + len('*** START OF THE PROJECT GUTENBERG EBOOK A TALE OF TWO CITIES ***'):]

    end = data.find('*** END OF THE PROJECT GUTENBERG EBOOK A TALE OF TWO CITIES ***')
    if end != -1:
        data = data[:end]

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
