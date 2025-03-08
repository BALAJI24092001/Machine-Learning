# Large Language Models

An LLM, a large language model, is a neural network designed to understand, generate, and respond to human-like text. These models are deep neural networks trained on massive amounts of text data, sometimes encompassing large portions of the entire publicly available text on the internet.

LLMs utilize an architecture called the transformer, which allows them to pay selective attention to different parts of the input when making predictions, making them especially adept at handling the nuances and complexities of human language.

<center>
<img src='../../assets/LLM training.png'/>
</center>

After obtaining a pretrained LLM from training on large text datasets, where the LLM is trained to predict the next word in the text, we can further train the LLM on labeled data, also known as finetuning.

The two most popular categories of finetuning LLMs include instructionfinetuning and finetuning for classification tasks. In instruction-finetuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. In classification finetuning, the labeled dataset consists of texts and associated class labels, for example, emails associated with spam and non-spam labels.

Later variants of the transformer architecture, such as the so-called BERT (short for bidirectional encoder representations from transformers) and the various GPT models (short for generative pretrained transformers), built on this concept to adapt this architecture for different tasks

## Resources

1. [Tokenizer](https://tiktokenizer.vercel.app/)
2. [OpenAI Models](https://platform.openai.com/docs/models/)

## Leaderboards

1. [Chatbot Arena LLM Leaderboard](https://lmarena.ai/)
2. [Seal](https://scale.com/leaderboard)

## Blogs

1. [A Practical Introduction to LLMs - Shaw Talebi](https://medium.com/towards-data-science/a-practical-introduction-to-llms-65194dda1148)
2. [Cracking Open the OpenAI (Python) API - Shaw Talebi](https://medium.com/towards-data-science/cracking-open-the-openai-python-api-230e4cae7971)
