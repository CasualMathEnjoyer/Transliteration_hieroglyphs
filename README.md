# Transliteration of Hieroglyphs Using Neural Networks

This repository contains Python code for a Bachelor Thesis Project: **Sequence Translations and Their Applications**.

This repository captures the state of the project at the time of writing the Bachelor's thesis (2024).

## Structure of the Repository

- **1_encoder_only_transformer**: Code and models for an encoder-only transformer architecture.
- **2_en_de_transformer**: Code and models for an encoder-decoder transformer architecture.
- **2_en_de_LSTM**: Code and models for an encoder-decoder LSTM architecture.
- **data**: Directory containing the datasets used for training and evaluation.
- **gensim_visualization**: Scripts and notebooks for visualizing embeddings using Gensim.
- **metrics_evaluation**: Scripts for evaluating the performance of the models using various metrics.
- **plot_model**: Scripts and tools for plotting model architectures and attention maps.
- **trained_models**: Pre-trained models saved for inference and further analysis.
- **requirements.txt**: List of dependencies required to run the project.

## Prerequisites

Ensure you have Python 3.x installed. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Pre-trained Models

The `trained_models` directory contains the models mentioned in my thesis.
The models can be loaded and used for inference or further training.

## Acknowledgments

This project was developed as part of a Bachelor Thesis at CVUT FJFI by Katka Morovicsova.