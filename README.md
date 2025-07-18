---
license: gpl-3.0
datasets:
- pieetie/pubmed-abstract-summary
language:
- en
base_model:
- google/flan-t5-small
pipeline_tag: summarization
library_name: transformers
---
# FLAN-T5-Small Fine-Tuned for Summarization
This model is a fine-tuned `flan-t5-small` transformer trained to summarize text when provided the instructions to do so. The model was trained on abstracts from publically available research papers and their summaries.

## Model Overview
* **Base Model:** `flan-t5-small`
* **Type:** Text summarization
* **Framework:** Hugging Face Transformers
* **Training Method:** LoRA using the PyTorch backend

## Dataset Used
**pieetie/pubmed-abstract-summary**
* 4,331 biomedical research abstracts and their one-sentence summaries from the American National Library of Medicine.
* Summaries contain the key findings, methods, and significance of the research.
* Entries were duplicated and the duplicated entries' abstracts had their sentences shuffled around.

## Training Configuration

### LoRA Parameters
| Parameter      | Value |
| :------------: | :---: |
| Rank           | 16    |
| $`\alpha`$     | 32    |
| Dropout        | 0.05  |
| Bias           | none  |
| Target Modules | q, v  |

### Training Parameters
| Parameter        | Value  |
| :--------------: | :----: |
| Epochs           | 5      |
| Learning Rate    | 1e-3   |
| Batch Size       | 16     |
| Optimizer        | AdamW  |
| Scheduler        | Linear |
| Padding Token ID | -100   |

## Evaluation Metrics
| Metric          | Value       |
| :-------------: | :---------: |
| Training Loss   | ~1.7974     |
| Validation Loss | ~1.4674     |
| Training Time   | ~1.2918 hrs |

## Hardware
**M4 Macbook Pro**
* 16 GB Unified RAM
* 10-core CPU
* 10-core GPU

## Example Inference
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

summarizer = pipeline(
    task="summarization",
    model=AutoModelForSeq2SeqLM.from_pretrained("KedarPanchal/flan-t5-small-summary-finetune")
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small")
)

text = """Summarize the following: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""

summarizer(text, minnew_tokens=10, do_sample=False)
```
> **Generated text:** The transformer, based on attention mechanisms dispensing with recurrence and convolutions, achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over existing best results, and establishing a single-model state-of-the-art BLUE score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the literature training costs of best models.

## License
The GNU GPLv3 License 2025 - [Kedar Panchal](https://huggingface.co/KedarPanchal). Please look at the [LICENSE](LICENSE) for more information.