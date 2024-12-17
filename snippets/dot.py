from mechir import Dot
from mechir.data import MechIRDataset, DotDataCollator
from mechir.perturb import perturbation
from mechir.plotting import plot_components
import torch
from torch.utils.data import DataLoader


@perturbation
def relevant_perturbation(text: str) -> str:
    return text + 'relevant'


model = Dot('tasb')
dataset = MechIRDataset('msmarco-passage/trec-dl-2019/judged')
collator = DotDataCollator(model.tokenizer, transformation_func=relevant_perturbation)
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)

patching_head_outputs = []
for batch in loader:
    output = model.patch(**batch, patch_type='head_all')
    patching_head_outputs.append(output)

mean_head_outputs = torch.mean(torch.stack(patching_head_outputs), axis=0)

plot_components(mean_head_outputs.detach().to("cpu").numpy())
