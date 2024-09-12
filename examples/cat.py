from mechir import Cat, PairDataset, CatDataCollator, perturbation
from torch.data import DataLoader
import pandas as pd
import json

@perturbation
def perturb(self, text : str, query : str = None) -> str:
    return text + "Apple"

def patch_cat(model_name_or_path : str, out_path : str, dataset : str, num_labels : int = 2, batch_size : int = 1, patch_type : str = 'block_all', block_list : str = None, pair_path : str = None):
    
    pairs = pd.read_csv(pair_path) if pair_path is not None else None
    
    patch_args = {
        'patch_type' : patch_type,
    }
    
    if "by_pos" in patch_type:
        assert block_list is not None, "block_list must be provided for by_pos patching"
        patch_args['block_list'] = block_list

    model = Cat(model_name_or_path, num_labels)
    dataset = PairDataset(dataset, pairs)
    data_collator = CatDataCollator(model.tokenizer, perturb)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    result = {
        'scores' : [],
        'scores_p' : [],
        'patch_output' : [],
    }

    for batch in dataloader:
        out = model(**batch, **patch_args)
        result['scores'].append(out['scores'].numpy())
        result['scores_p'].append(out['scores_p'].numpy())
        result['patch_output'].append(out['result'].numpy())
    
    # dump dict of numpy to json
    with open(out_path, 'w') as f:
        json.dump(result, f)


    
