# MechIR

## An IR Library for Mechanistic Interpretability

This library exists to make encoders first-class citizens in Mechanistic Interpretability, particularly with a focus on Information Retrieval and representation learning. We presented this work as a demo at ECIR 2025: https://arxiv.org/pdf/2501.10165

We have done our best to support many of the features present in [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens/tree/main). Additionally, we provide extensive functionality for activation patching (more details in our demo!). We are in the process of producing SAEs for different common IR models!.


Heavily inspired by the paper **Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models** by Catherine Chen, Jack Merullo, and Carsten Eickhoff (Brown University, University of Tübingen), their original code can be found [here](https://github.com/catherineschen/axiomatic-ir-interventions/tree/main)

## Demonstration

Primary files can be found in the /notebooks folder. If you use our live versions and want to run our experiments make sure to choose a GPU instance of Colab. You can easily change our notebook to observe different behaviour so try your own experiments!

* experiment.ipynb: This notebook demonstrates how to use the MechIR library to perform activation patching on a simple neural retrieval model. Here is a live version [on Colab](https://drive.google.com/file/d/1ZR1ZD2bwcLh5mQG_x561PCpydHK4mhap/view?usp=sharing)
* activation_patching_considerations.ipynb: This notebook provides a more in-depth look at the activation patching process and the considerations that go into it. Here is a live version [on Colab](https://drive.google.com/file/d/1TSRCvixMo4YPRwDqsjf7kuci6fGwwQG2/view?usp=sharing)

## Installation

### Latest Release (Unstable)
```
pip install git+https://github.com/Parry-Parry/MechIR.git
```

### PyPI (Stable)
```
pip install mechir
```

## Usage

### Models 

Currently we support common bi- and cross-encoder models for neural retrieval. The following models are available:

* Dot: A simple dot-product model allowing multiple forms of pooling
* Cat: A cross-encoder model with joint query-document embeddings and a linear classification head
* MonoT5: A sequence-to-sequence cross-encoder model based on T5

To load a model, for example TAS-B, you can use the following code:

```python
from mechir import Dot

model = Dot('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco')
```

### Datasets and Perturbations 

To probe model behaviour we need queries and documents, we retrieve these from ir-datasets though you can use your own (MechDataset). To load an IR dataset, for example MS MARCO, you can use the following code:

```python
from mechir import MechIRDataset

dataset = MechIRDataset('msmarco-passage/dev')
```

The second step of probing is to create a perturbation of text to observe how model behaviour changes, we can do this simply with the perturbation decorator:

```python
from mechir.perturb import perturbation

@perturbation
def my_perturbation(text):
    return text + "MechIR"
```

We can then apply this perturbation efficiently using our dataset and a torch dataloader

```python
from torch.utils.data import DataLoader
from mechir.data import DotDataCollator

collate_fn = DotDataCollator(model.tokenizer, transformation_func=my_perturbation)
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=8)
```

### Activation Patching

Activation patching is a method to isolate model behaviour to particular components of the model commonly attention heads. There are several ways to perform activation patching, a simple case is to patch all heads:

```python
patch_output = []
for batch in dataloader:
    patch_output.append(model.patch(**batch, patch_type="head_all"))

patch_output = torch.mean(torch.stack(patch_output), axis=0)
```

We can then easily visualise the attention heads which activate strongly for our perturbation:

```python
from mechir.plotting import plot_components

plot_components(patch_output)
```

## Citation
If you use our work in research please let us know! We will showcase the work in this repository to encourage more encoder interpretability. Please use the following citation if you find our work useful!
```
@inproceedings{mechir:2025,
  author       = {Andrew Parry and
                  Catherine Chen and
                  Carsten Eickhoff and
                  Sean MacAvaney},
  title        = {MechIR: {A} Mechanistic Interpretability Framework for Information
                  Retrieval},
  booktitle    = {Advances in Information Retrieval - 47th European Conference on Information
                  Retrieval, {ECIR} 2025, Lucca, Italy, April 6-10, 2025, Proceedings,
                  Part {V}},
  series       = {Lecture Notes in Computer Science},
  volume       = {15576},
  pages        = {89--95},
  publisher    = {Springer},
  year         = {2025},
  url          = {https://doi.org/10.1007/978-3-031-88720-8\_16},
  doi          = {10.1007/978-3-031-88720-8\_16},
  biburl       = {https://dblp.org/rec/conf/ecir/ParryCEM25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
