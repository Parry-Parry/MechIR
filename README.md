# MechIR

## An IR Library for Mechanistic Interpretability

Heavily inspired by the paper **Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models** by Catherine Chen, Jack Merullo, and Carsten Eickhoff (Brown University, University of Tübingen), their original code can be found [here](https://github.com/catherineschen/axiomatic-ir-interventions/tree/main)

## Demonstration

Primary files can be found in the /notebooks folder. If you use our live versions and want to run our experiments make sure to choose a GPU instance of Colab. You can easily change our notebook to observe different behaviour so try your own experiments!

* experiment.ipynb: This notebook demonstrates how to use the MechIR library to perform activation patching on a simple neural retrieval model. Here is a live version [on Colab](https://drive.google.com/file/d/1L34CPsgKSW8akHOet15j_nuuXNINX7gt/view?usp=sharing)
* activation_patching_considerations.ipynb: This notebook provides a more in-depth look at the activation patching process and the considerations that go into it. Here is a live version [on Colab](https://drive.google.com/file/d/1TSRCvixMo4YPRwDqsjf7kuci6fGwwQG2/view?usp=sharing)