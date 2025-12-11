# Sentiment Analysis on the Sentiment140 Dataset  
### Comparison of GRU, TextCNN, and DistilBERT Models

This project performs binary sentiment classification (positive vs negative) on the Sentiment140 dataset using three different deep learning architectures:

1. GRU (Gated Recurrent Unit)
2. TextCNN
3. DistilBERT (Transformer-based model)

The goal is to evaluate how traditional neural architectures compare against transformer-based models on a large-scale Twitter sentiment dataset.

---

## 1. Dataset Information

The dataset contains 1.6 million tweets labeled as:
- 0 → Negative  
- 4 → Positive (mapped to 1 in this project)

Source:  
Stanford NLP Group  
https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  

Files used:
- `training.1600000.processed.noemoticon.csv`

---

## 2. Preprocessing Steps

- Lowercasing  
- Removing URLs, mentions, and special characters  
- Removing stopwords  
- Lemmatization  
- Tokenization (for GRU/CNN)  
- DistilBERT tokenizer encoding (for transformer model)

---

## 3. Models Implemented

### GRU Model
- Embedding layer
- GRU(64) unit
- Dense(1, sigmoid)
- Binary cross-entropy loss

### TextCNN Model
- Embedding layer
- Conv1D(128, kernel size 5)
- GlobalMaxPooling
- Dense(1, sigmoid)

### DistilBERT Model
- Pretrained DistilBERT encoder
- Classification head (2 classes)
- Fine-tuned for sentiment classification

---

## 4. Evaluation Metrics

Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Accuracy comparison chart  

The following table shows a sample evaluation output:

| Model       | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| GRU         | 0.7250   | 0.7237    | 0.7356 | 0.7296   |
| TextCNN     | 0.7130   | 0.6927    | 0.7746 | 0.7313   |
| DistilBERT  | 0.6427   | 0.7078    | 0.4964 | 0.5835   |

---

## 5. Visualizations

The notebook includes:
- Accuracy comparison bar chart  
- Confusion matrices for all models  
- Metrics summary table  

These visuals help compare classical sequence models vs transformer-based models.

---

## 6. File Structure

Sentiment140-NLP-Models

- notebook.ipynb 
- model_comparison.csv 
- README.md 
- requirements.txt 

---

## 7. How to Run the Notebook

Open in Google Colab:

1. Upload `notebook.ipynb` to Google Colab  
2. Run all cells  
3. The dataset is downloaded automatically using a direct Stanford URL  
4. All three models train and evaluate  

---

## 8. Requirements

Add this to `requirements.txt` if running locally:

pandas
numpy
tensorflow
torch
transformers
scikit-learn
matplotlib
seaborn
wordcloud

---

## 9. Conclusion

This project highlights:
- GRU and TextCNN perform strongly despite their simplicity.
- DistilBERT requires more training steps to outperform classical architectures.
- Preliminary fine-tuning shows transformer models need more training time but have higher potential.

Future improvements:
- Full fine-tuning of DistilBERT (3–5 epochs)
- Use RoBERTa or BERT-large
- Hyperparameter tuning
- Streamlit deployment

