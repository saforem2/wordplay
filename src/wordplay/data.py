"""
data.py
"""
from __future__ import absolute_import, annotations, division, print_function
import os
import numpy as np
import tiktoken
import datasets
import tqdm
# from datasets import load_dataset
from typing import Optional

from pathlib import Path
# import wordplay as wp
from wordplay.configs import HF_DATASETS_CACHE_DIR


HERE = Path(os.path.abspath(__file__)).parent
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", None)
if HF_DATASETS_CACHE is None:
    # HF_DATASETS_CACHE = HERE / ".cache" / "huggingface"
    HF_DATASETS_CACHE = HF_DATASETS_CACHE_DIR.as_posix()
else:
    print(f'Caught {HF_DATASETS_CACHE=} from env')

HF_DATASETS_CACHE = Path(str(HF_DATASETS_CACHE))
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE.as_posix()
print(f'Using {HF_DATASETS_CACHE.as_posix()=}')

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
# num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW
# speed.
# it is better than 1 usually though
# num_proc_load_dataset = num_proc
# HF_DATASETS_CACHE="./cache"


def get_dataset(
        name: str,
        num_proc: Optional[int] = None,
):
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = datasets.load_dataset(
        name,
        num_proc=num_proc
    )
    # owt by default only contains the 'train' split, so create a test split
    # split_dataset = dataset["train"].train_test_split(
    #     test_size=0.0005,
    #     seed=2357,
    #     shuffle=True
    # )
    # # rename the test split to val
    # split_dataset['val'] = split_dataset.pop('test')
    return dataset


# def split_dataset(dataset):  # type:ignore
#     pass


def tokenize(
        dataset: datasets.Dataset,
        data_dir: os.PathLike,
        encoding: str = 'gpt2',
        num_proc: int = 8,
):
    # we now want to tokenize the dataset. first define the encoding function
    # (gpt2 bpe)
    enc = tiktoken.get_encoding(encoding)

    def process(example):
        # encode_ordinary ignores any special tokens
        ids = enc.encode_ordinary(example['text'])
        # add the end of text token, e.g. 50256 for gpt2 bpe
        ids.append(enc.eot_token)
        # note: I think eot should be prepended not appended... hmm. it's
        # called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use
    # for training
    outdir = Path(data_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    print(f'Saving splits to: {outdir.as_posix()}')
    for split, dset in tokenized.items():  # type:ignore
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(outdir.as_posix(), f'{split}.bin')
        # (can do since enc.max_token_value == 50256 is < 2**16)
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(
                range(total_batches),
                desc=f'writing {filename}'
        ):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
