# What is the environmental impact of the models used on Active Tigger?

Using AI models come with a consumption of energy and resources that are important to consider. This page discusses the environmental impact of models.

## The environmental impact of AI models

Environmental impacts of AI raise [important concerns](https://www.nytimes.com/2024/08/26/climate/ai-planet-climate-change.html) and became an [important issue for research](https://arxiv.org/abs/2111.00364).

For NLP, the energy consumption can varie greatly depending on the **type of technology used**, **the energy origin used to run the servers** but also on the **the methodology chosen** to evaluate the impacts.

Active Tigger is designed to be frugal. It uses **BERT** models, which are small, and then **less energy-intensive than large generative models**. However, it is still important to understand the environmental impact of these models and the methodologies used to evaluate them.

## When does a model consume energy ?

There are 3 main phases in the life cycle of a model that can consume energy and have an environmental impact : the **training phase**, the **fine-tuning phase**, and the **inference phase**.

### ⚙️ Training phase

This phase involve using a huge text dataset to train a model. **Several studies show that generative models (such as GPT) consume vast resources at this stage**. [An analysis of the GPT-3 model](https://arxiv.org/abs/2104.10350), which contains 175 billion parameters, estimated it generated 552.1 tonnes of CO₂.

By contrast, **models smaller such as BERT (~100 millions parameters) consume less energy**. [One research team](https://arxiv.org/abs/2311.10267) trained several BERT models using different configurations (hardware, batch size, sequence length). The environmental cost ranged from 58.9 kg of CO₂ for 124.1 kWh to 199.1 kg of CO₂ for 419.6 kWh.

[Another study](https://arxiv.org/abs/1906.02243) measured 1,438 kg of CO₂ for training a BERT base model on 64 V100 GPUs — **showing how emissions depend heavily on the hardware used and the data volume processed**.

### 🧪 Fine-tuning phase

Fine-tuning involves adapting a pre-trained language model to a specific task. **This phase is less energy-intensive than the initial training**. Nevertheless, it is performed far more frequently than the initial training.

[One study](https://arxiv.org/abs/2311.10267) showed that energy consumption during fine-tuning depends mainly on the total number of tokens processed (rather than their individual size) and the wall clock time.

### 🔍 Inference phase

Inference is the phase where the model is used to make predictions or generate text based on new inputs. It is generally less energy-intensive than training, but it can still have a significant environmental impact, especially when the model is used frequently or at scale.

In the case of research, models are not scaled for millions of users, so their use is less energy-intensive than general public applications. However, **the energy consumption of inference can still be significant**, especially for large models or when processing large volumes of data.

## Going further: how to evaluate the environmental impact of AI models?

### The energy consumption of models

To evaluate a model’s impact, it is possible to **examine the energy consumed by the infrastructure used during training**. Result depends in particular on [the country](https://arxiv.org/abs/2211.02001) — and therefore its carbon mix — where the servers are located. For example, while training the Bloom model, using the Jean-Zay supercomputer in France, required 433 MWh of electricity (compared to 324 MWh for the OPT model), Bloom’s training generated 25 tonnes of CO₂, versus 70 tonnes for OPT.

In addition to that, **the reported environmental impact depends on the type of energy reported by companies**. Some companies purchase green certificates, [which neither guarantee actual renewable electricity consumption nor the funding of new infrastructure](https://www.nature.com/articles/s41558-022-01379-5?). Others opt for PPA contracts, which directly finance renewable energy production through long-term agreements with producers.

Evaluating the energy consumption of models is complicated. For instance, **most studies do not account for all the steps preceding the final model training**, such as experiments on intermediate versions, so-called “ablation” tests (removing parts of the model to test their importance), or the many adjustments needed to reach the final version. [Some authors](https://arxiv.org/abs/2211.02001)  estimate that these stages alone can double the total carbon footprint of a project.

Most studies that evaluate the environmental impact of language models (LLMs) only consider part of the problem. [As some authors point out](https://arxiv.org/abs/2110.11822), **it is essential to conduct a full life cycle assessment (LCA)**. This means considering not only the energy used during training, but also the pollution linked to the manufacture of the required infrastructure (servers, storage devices, etc.), from the extraction of raw materials to end-of-life disposal.

This includes two often overlooked components:

- Embodied emissions: all the pollution generated upstream (manufacturing, transport, installation of equipment);

- Idle consumption: electricity used by servers even when not actively in use.

As an example, the carbon footprint [associated with training](https://arxiv.org/abs/2211.02001) the Bloom model **rises from 25 to nearly 50 tonnes of CO₂** when all these parameters are included.

Furthermore, **it is important not to limit impact assessments to carbon emissions alone**. One must consider all forms of pollution generated. For instance, the manufacture of chips used to train and run models requires vast quantities of water — another major ecological cost that is often overlooked.

To assess the environmental impact of deep learning models, one must also **take into account the energy spent on fine-tuning the models and running them during inference**. While hard to estimate, [some studies](https://arxiv.org/abs/2104.10350) suggest that inference (using the model) accounts for the vast majority of energy consumption in large models (especially GPT-type).


## A few additional remarks

At first glance, using models like BERT — whether for training, fine-tuning, or inference — appears less energy-intensive than using large generative models like GPT. However, **even if these models consume relatively less, this is still important to keep in mind their potential impacts**, especially if the number of users increases.

As we have seen, current impact assessment methodologies **do not account for all pollution sources**. It is therefore essential to minimize unnecessary energy use and **consistently question the relevance and necessity of each project in light of the energy it consumes**.

Furthermore, even in countries with a largely decarbonized energy mix, **electricity remains a limited resource** — one that could be allocated to other equally or more essential needs. **A low-carbon mix does not justify irresponsible use, nor should it lead to forgetting the principle of digital sobriety**.