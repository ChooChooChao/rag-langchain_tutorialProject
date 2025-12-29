# rag-langchain_tutorialProject

This repository is a **learning project** created by following along with a YouTube tutorial.

## 📺 Source
- Tutorial: *"RAG + Langchain Python Project: Easy AI/Chat For Your Doc"*  
- Creator: *pixegami*  
- Link: https://www.youtube.com/watch?v=tcqEUSNCn8I&list=WL

All credit for the original project concept and structure goes to the creator.

## 🎯 Purpose of This Repo
This repository exists to:
- Follow along with the tutorial step by step
- Reinforce my understanding of the concepts
- Experiment with modifications and extensions
- Serve as a reference for future projects

## 🔧 My Additions / Changes
- Switched from OpenAI embeddings to local HuggingFace embeddings
- Adapted setup for my local environment (macOS / Conda)

## 📝 Notes
This project is **not an original work**. It is a personal learning exercise based on the tutorial above.

## Results
- Using the HuggingFace model generated significantly worse responses to prompts compared to OpenAI model seen in the video.
  - Somethings that could be tweaked to improve this include the model_id, or other hyperparamters like k, chunk_size, or overlap
