# Advanced Next-Word Prediction Model

This repository contains a state-of-the-art next-word prediction model built using PyTorch, leveraging a transformer decoder architecture with advanced features like pretrained GloVe embeddings, subword tokenization, and beam search decoding. The model is designed for scalability and can be trained on large datasets for production-level performance.

## Features

- **Transformer Decoder Architecture**: Autoregressive modeling with causal masking for natural sequence generation.
- **Pretrained GloVe Embeddings**: Integrates 300-dimensional GloVe embeddings for richer semantic representations.
- **Subword Tokenization**: Uses SentencePiece with Byte Pair Encoding (BPE) to handle out-of-vocabulary words.
- **Mixed Precision Training**: Employs FP16 training with `torch.cuda.amp` for faster computation and reduced memory usage.
- **Learning Rate Warmup**: Linear warmup schedule for stable transformer training.
- **Dynamic Padding & Batching**: Efficiently handles variable-length sequences with custom collation.
- **Checkpointing**: Saves the best model based on validation loss for resuming training.
- **Evaluation Metrics**: Computes perplexity, top-1, and top-5 accuracy on a validation set.
- **Beam Search Decoding**: Generates coherent predictions with configurable beam width.

## Dataset

The model is designed to scale with large corpora. For production, we recommend the following datasets:

- **WikiText-103**: A large-scale dataset for language modeling, available via [Hugging Face Datasets](https://huggingface.co/datasets/wikitext).
- **BooksCorpus**: A diverse corpus of books, accessible through [Hugging Face Datasets](https://huggingface.co/datasets/bookcorpus) or custom scraping.
- **CommonCrawl**: Large-scale web data, available at [CommonCrawl](https://commoncrawl.org/).

For demonstration, the code includes a small sample dataset. Replace it with one of the above for better generalization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/next-word-predictor.git
   cd next-word-predictor
   ```

2. Install dependencies:
   ```bash
   pip install torch sentencepiece numpy
   ```

3. Download GloVe embeddings:
   - Download `glove.6B.300d.txt` from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/).
   - Place the file in the project root or update the `glove_path` in `advanced_next_word_prediction_model.py`.

4. (Optional) Install SentencePiece for subword tokenization:
   ```bash
   pip install sentencepiece
   ```

5. (Optional) For GPU support, ensure CUDA is installed and compatible with your PyTorch version.

## Usage

1. **Prepare the Dataset**:
   - Replace `sample_texts` and `val_texts` in `advanced_next_word_prediction_model.py` with your dataset (e.g., WikiText-103).
   - Train a SentencePiece model on your corpus:
     ```python
     vocab = SPVocabulary(vocab_size=10000)
     vocab.train(your_texts)
     ```

2. **Train the Model**:
   ```bash
   python advanced_next_word_prediction_model.py
   ```
   - The model will train for 10 epochs (configurable) and save checkpoints to the `checkpoints/` directory.
   - Training progress, validation perplexity, and top-k accuracy are printed per epoch.

3. **Inference**:
   - Use the `beam_search` function for predictions:
     ```python
     input_text = "The quick brown"
     predicted_sequence = beam_search(model, vocab, input_text, seq_length=20, device=device)
     print(f"Predicted Sequence: {predicted_sequence}")
     ```

4. **Example Output**:
   ```
   Input: The quick brown
   Predicted Sequence: The quick brown fox jumps over the lazy dog.
   ```

## Scaling the Model

- **Dataset**: Use large corpora like WikiText-103 or BooksCorpus. Preprocess with SentencePiece for a 50k+ token vocabulary.
- **Model Size**: Increase `D_MODEL` (e.g., 768), `NHEAD` (e.g., 12), `NUM_LAYERS` (e.g., 12), and `DIM_FEEDFORWARD` (e.g., 3072).
- **Training**: Use distributed training with PyTorch DDP or DeepSpeed on multi-GPU/TPU clusters. Train for 50-100 epochs.
- **Hardware**: Leverage AWS/GCP with A100 GPUs or TPUs for large-scale training.
- **Optimization**:
  - Apply quantization using `torch.quantization` for faster inference.
  - Export to ONNX for deployment: `torch.onnx.export(model, ...)`.
  - Deploy as a REST API using FastAPI or Flask.

## Evaluation

- **Metrics**: The model evaluates perplexity, top-1, and top-5 accuracy on the validation set.
- **Perplexity**: Measures the model's uncertainty in predictions (lower is better).
- **Top-k Accuracy**: Checks if the correct next word is in the top-1 or top-5 predictions.
- **BLEU**: For sequence generation, compute BLEU scores against reference texts using libraries like `nltk` or `sacrebleu`.

## Future Enhancements

- **Multi-task Learning**: Add objectives like masked language modeling or sequence classification for better generalization.
- **Advanced Architectures**: Integrate TransformerXL or Longformer for longer context modeling.
- **Self-Supervised Pretraining**: Pretrain on a large corpus, then fine-tune for next-word prediction.
- **Attention Visualization**: Use `bertviz` to visualize attention weights for interpretability.


