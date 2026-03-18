# David vs. Goliath - Final Year Project

This repository will document and reflect the progress of development of my final year project, a _Small Language Model_ (SLM) filter to protect _RAG-based_ Large Language Models (LLM) from _Indirect Prompt Injection_ (IPI).

The dataset to be used will be Microsoft's Benchmark for Indirect Prompt Injection Attacks (BIPIA) due to its open sourceness, robustness and ease of access.

## Specs

- SLM: Microsoft Phi-3 Mini 4k Instruct, DistiliBERT

- LLM: Llama 3.1 8B Instruct

- VDB: ChromaDB

- Dataset(s): Microsoft BIPIA

## Planned Features

17/01/2026 - _Memory Swapping_. The laptop in which this project is being developed possesses only 8GB of VRAM. Phi-3 3.8B Mini Instruct has been observed to require around 2.11GB, and the Llama 8B Instruct 5.1 GB. To rigorously and effectively ensure that the laptop is capable of running both models "at the same time" (simulating an actual RAG system that contains both the LLM and the SLM filter):

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

23/01/2026 - _Implemented an example IPI vulnerability test_ on the Phi-3 model to demonstrate the effects of altering the prompt on the tokenizer and model. Phi-3 is used to demonstrate this vulnerability, where an example clean traceback call sourced from BIPIA, coupled with the given code, is then appended with a humanly-written string by a threat actor (simulating IPI) that attempts to diverge intended performance entirely. In ``vulnerability.py``, the prompt is appended with an adversarial phrase to ignore previous comments and instead supply the user with a comparison between two non-existent GPUs.

<img src="image/README/1769173029746.png" width=100% height=100% alt="image of a successful IPI attack"> </img>

10/02/2026 - _Automated IPI vulnerability test_ from the previously written function. Just automates the test rather than running it multiple times.

13/02/2025 - _SIMULATION of Agentic Model Vulnerabilities_. Gave the SLM a set of read and write tools as well as an agentic mindset in the form of a system prompt, and simulated an injection attack onto it by telling it to delete a file when the initial context was to fix a code based on a traceback call, for example. The model not only executed the adversarial phrase's instructions, but also kept it specifically in JSON format, completely ignoring the initial prompt.

<img src="image/README/1770933279701.png" width=60% height=30% alt="image of a successful IPI attack"> </img>

10/03/2026 - _Constructing the Dataset for the SLM filters._ Throughout the experimentation and testing of vulnerabilities with language models, both through adversarial phrases and tool calls, it has been concluded that there should be two minimal, lightweight but specialized filters that monitor both risky plaintext, as well as suspicious code.

13/03/2026 - _Test Training DistiliBERT to classify malignant and benign text using BIPIA_. For now, a selection of code tracebacks were taken from the BIPIA dataset. 50% of them were augmented with adversarial phrases, and a few benign but seemingly suspicious phrases were also added to the dataset. This maintains an approximate 50/50 even split with test data and prevents classification bias. DistiliBERT was selected for being an extremely lightweight model (less than 100M parameters) and thus will be a great benchmark in the event of using multiple SLMs to classify data. It is worth noting that a model as small as BERT would suffer _diminishing_ returns from using a technique such as QLoRA as the loss in accuracy is not worth the gain in speed, due to it already being lightweight.
![1773426205268](image/README/1773426205268.png)
Above shows the model undergoing a 3 epoch training. It can be observed that the eval_loss consistently drops in 3 epochs, demonstrating the capability of a tiny model to learn patterns when given a specialized task.
_ISSUE_: The BIPIA dataset doesn't offer enough data for DistiliBERT to learn properly.
![1773428238749](image/README/1773428238749.png)
The above image shows 3 different prompts, in differing severity according to the highlighted color, as well as the model's inference results. We can see that all of them are within the range of 0.5, showing a critical uncertainty within the model that is most probably as a result of a lack of data (BIPIA offers 50-100 total entries for the model to be trained on).

18/03/2026 - _Evaluating and Experimenting with various datasets_. The biggest issue the project is facing is the scarcity and variety of data. The filter is inadequately trained and requires careful work to help fine tune. Today, Deepset's prompt injection dataset was used to train the filter. The issue this time is that the dataset is consisted purely of plain English statements, and thus any statement that seems to deviate from this style is deemed unsafe. Will be moving forward by attempting to diversify the data between English and code-like statements in an equal split as to provide the filter enough knowledge.

- V1 - ~100 samples from BIPIA. Model was too unconfident and all inferences rested at a consistent 0.5-0.59 prediction between safe and unsafe.

- V2 - ~400 samples from Deepset's Prompt Injection dataset. Model was more confident in plain English statements but was clueless with code statements and assumed all were unsafe.

- V3 - ~600 samples combination of Deepset and BIPIA. Model is now under-relaxed, confident in code statements but unable to determine that injections are unsafe consistently.
