# Exploring the Heterogeneity of Tabular Data: A Diversity-aware Data Generator via LLMs

Welcome to **DATE**, a LLM-based tabular data generator for heterogenous data. 

Tabular data generation has become increasingly essential for enabling robust machine learning applications, which require large-scale, high-quality data.  Existing solutions leverage generative models to learn original data distributions. However, real-world data are naturally heterogeneous with diverse distributions, making it challenging to obtain a universally good model for diverse data generation. To address this limitation, we introduce Diversity-Aware Tabular data gEnerator (DATE), a framework that (i) prepares high-quality and distributionally distinct examples for in-context learning by effectively partitioning the original heterogeneous data into multiple diverse subsets; (ii) harnesses Large Language Models (LLMs) to explore the diversity of the partitioned distribution with decision tree reasoning as feedback, generating high-quality labeled data for each subset.

<img width="1075" height="535" alt="image" src="https://github.com/user-attachments/assets/a046ec45-f91a-459e-bfb2-371e949d04be" />

We present our **Source Code** and **Appendix** for users:
For detailed proofs and external experiments in the original paper, please read our [appendix](appendix.pdf).

To obtain the public data used in our experiment, please refer to the [IEEE Dataport](https://ieee-dataport.org/documents/heterogeneous-data-augmentation-benchmark).
We also public it on Huggingface as [Heterogenous_Synthesis_Benchmark](https://huggingface.co/datasets/CurryOvO/Heterogenous_Synthesis_Benchmark)

For running Module 1: DGR-based Prompt Designing, please refer to [ModelShare_with_DSR_final.py](ModelShare_with_DSR_final.py).

For running Module 2: Distribution-Specific Generation, please refer to [dataGeneration.py](dataGeneration.py).

Files with the `test_` prefix are test files. For example, (test_forward)[test_forward.py] is a testing file based on forward-greedy selection.

