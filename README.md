# David vs. Goliath - Final Year Project

This repository will document and reflect the progress of development of my final year project, a _Small Language Model_ (SLM) filter to protect _RAG-based_ Large Language Models (LLM) from _Indirect Prompt Injection_ (IPI).

The dataset to be used will be Microsoft's Benchmark for Indirect Prompt Injection Attacks (BIPIA) due to its open sourceness, robustness and ease of access.

## Specs

- SLM: Microsoft Phi-3 Mini 4k Instruct

- LLM: Llama 3.1 8B Instruct

- VDB: ChromaDB

- Dataset(s): Microsoft BIPIA

## Planned Features

17/01/2026 - _Memory Swapping_. The laptop in which this project is being developed possesses only 8GB of VRAM. Phi-3 3.8B Mini Instruct will require around 2.5GB, while the Llama 8B Instruct will need around 5.5 GB. To rigorously and effectively ensure that the laptop is capable of running both models "at the same time" (simulating an actual RAG system that contains both the LLM and the SLM filter):

- The SLM will first be loaded into memory to scan the data and produce a result (based on whether the data is malignant or benign)

- The result will then be stored and the SLM will then offload from memory (freeing 2.5GB from the 8GB of VRAM)

- The LLM will then be loaded (5.5GB into VRAM) and process the result accordingly, producing the output.

By following this procedure, approximately a net total of 2.5GB VRAM will be saved during runtime execution, by concurrently swapping between language model operation intelligently. Whether this has an effect on final latency will be observed during the examination stages of the model.

---

18/01/2026 - _Quantized Low-Rank Adaptation_ (QLoRA). 4-bit Quantization (NormalFloat 4) is the method to effectively fit language models into a more constrained VRAM environment with minimal performance degradation. Weights in these language models are usually stored in 16 bits. With NF4, they will be stored into 4 bits instead ($2^4 = 16$). This is a _necessary_ step as full parameter fine tuning or even loading the model without quantization will cost much more than 8GB of VRAM, rendering the project computationally infeasible to develop. However, with the model loaded via NF4, it will be infeasible to train the model, as the weights have been _frozen_ and will not be able to change. Therefore, QLoRA will be used to attach an amount of 16-bit matrices called 'Adapters' into the model, which will do the learning during the training process. These adapters are not only sleeker and more streamlined (a few megabytes in comparison to tens of gigabytes) but they are _portable_ (meaning you can use these trained LoRA adapters with other identical models, without altering the model itself), more stable because the base model remains untouched, thereby preventing _catastrophic forgetting_ (a fine tuning phenomenon), and are faster to train due to having fewer total parameters to update during the backwards pass.

## Project Log

17/01/2026 - Initialized the project. Created a script that checks current workstation specs (mainly utility purposes). Created utils.py which contains a memory offloader function for now, will be used for _Memory Swapping_.

---

18/01/2026 - Cloned BIPIA repo to be used for training the SLM. Acquired access to Llama 3.1. Updated memory swapping function and added authentication script to ensure access to both LLM and SLM before development.

---

23/01/2026 - _Implemented an example IPI vulnerability test_ on the Phi-3 model to demonstrate the effects of altering the prompt on the tokenizer and model. Phi-3 is used to demonstrate this vulnerability, where an example clean traceback call sourced from BIPIA, coupled with the given code, is then appended with a humanly-written string by a threat actor (simulating IPI) that attempts to diverge intended performance entirely. In ``vulnerability.py``, the prompt is appended with an adversarial phrase to ignore previous comments and instead supply the user with a double chocolate cookie recipe, all in full caps.

<img src="image/README/1769173029746.png" width=100% height=100% alt="image of a successful IPI attack"> </img>
