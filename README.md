# Learning on Model Weights using Tree Experts
Official PyTorch Implementation for the "Learning on Model Weights using Tree Experts" paper.  
<p align="center">
    🌐 <a href="https://horwitz.ai/probex/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2410.13569" target="_blank">Paper</a> 
</p>

___

> **Learning on Model Weights using Tree Experts**<br>
> Eliahu Horwitz*, Bar Cavia*, Jonathan Kahana*, Yedid Hoshen<br>
> <a href="https://arxiv.org/abs/2410.13569" target="_blank">https://arxiv.org/abs/2410.13569 </a> <br>
>
>**Abstract:** The increasing availability of public models begs the question: can we train neural networks 
> that use other networks as input? Such models allow us to study different aspects of a given neural network, 
> for example, determining the categories in a model's training dataset. However, machine learning on model weights
> is challenging as they often exhibit significant variation unrelated to the models' semantic properties (nuisance variation). 
> Here, we identify a key property of real-world models: most public models belong to a small set of *Model Trees*, where all 
> models within a tree are fine-tuned from a common ancestor (e.g., a foundation model). Importantly, we find that within 
> each tree there is less nuisance variation between models. Concretely, while learning across Model Trees requires complex 
> architectures, even a linear classifier trained on a single model layer often works within trees. While effective, these linear 
> classifiers are computationally expensive, especially when dealing with larger models that have many parameters. To address this, 
> we introduce *Probing Experts* (ProbeX), a theoretically motivated and lightweight method. Notably, ProbeX is the first probing 
> method specifically designed to learn from the weights of a single hidden model layer. We demonstrate the effectiveness of ProbeX by 
> predicting the categories in a model's training dataset based only on its weights. Excitingly, ProbeX can also map the 
> weights of Stable Diffusion into a shared weight-language embedding space, enabling zero-shot model classification.

___
**The code will be published in the coming days...**
