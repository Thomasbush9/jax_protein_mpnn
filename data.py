import time
import json
from torch.utils.data import Dataset


class StructureDataset(Dataset):
    def __init__(
        self,
        entries,
        verbose=True,
        truncate=None,
        max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWY',
    ) -> None:
        alphabet_set = set(alphabet)
        discard_count = {'bad_chars': 0, 'too_long': 0}

        self.data = []
        start = time.time()
        for i, entry in enumerate(entries):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set(seq).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(seq) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                if verbose:
                    print(name, bad_chars, seq)
                discard_count['bad_chars'] += 1

            if truncate is not None and len(self.data) == truncate:
                break

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))

        if verbose:
            print('discarded', discard_count)

    @classmethod
    def from_jsonl(cls, json_file, **kwargs):
        with open(json_file) as f:
            entries = [json.loads(line) for line in f]
        return cls(entries, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
