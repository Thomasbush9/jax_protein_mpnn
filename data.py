import jax 
import time
import numpy 
import jax.numpy as jnp 
import json 



#TODO: improve it by separate the init functions + torch framework
class StructureDataset():
    def __init__(
        self, 
        json_file, 
        verbose=True,
        truncate=None,
        max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWY',
    )-> None:
    
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }
        
        with open(json_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # start conversion of raw coords -> arrays 

                # check if in alphabet: 
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1 
                else:
                    if verbose:
                        print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] +=1

                if truncate is not None and len(self.data) == truncate:
                    return 
                
                if verbose and (i+1) % 1000 == 0:
                    elapsed = time.time() - start 
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))
                if verbose:
                    print('discarded', discard_count)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



