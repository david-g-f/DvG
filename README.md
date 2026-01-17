# David vs. Goliath - Final Year Project

This repository will document and reflect the progress of development of my final year project, a _Small Language Model_ (SLM) filter to protect _RAG-based_ Large Language Models (LLM) from _Indirect Prompt Injection_ (IPI).

## Planned Features

17/01/2026 - _Memory Swapping_. The laptop in which this project is being developed possesses only 8GB of VRAM. Phi-3 3.8B Mini Instruct requires around 2.5GB, while the Llama 8B Instruct will need around 5.5 GB. To rigorously and effectively ensure that the laptop is capable of running both models "at the same time" (simulating an actual RAG system that contains both the LLM and the SLM filter):

- The SLM will first be loaded into memory to scan the data and produce a result (based on whether the data is malignant or benign)

- The result will then be stored and the SLM will then offload from memory (freeing 2.5GB from the 8GB of VRAM)

- The LLM will then be loaded (5.5GB into VRAM) and process the result accordingly, producing the output.

By following this procedure, approximately a net total of 2.5GB VRAM will be saved during runtime execution, by concurrently swapping between language model operation intelligently. Whether this has an effect on final latency will be observed during the examination stages of the model.

## Project Log

17/01/2026 - Initialized the project. Created a script that checks current workstation specs (mainly utility purposes). Created utils.py which contains a memory offloader function for now, will be used for _Memory Swapping_.
