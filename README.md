# Turkish Sentiment Analysis (Amazon Reviews - Keras LSTM + BERT) 

This project is a desktop application for Turkish sentiment analysis.It runs two NLP approaches on the same input and displays results side-by-side:
* A custom-trained Keras LSTM model (trainable for your domain
* A pretrained BERT Transformer (savasy/bert-base-turkish-sentiment-cased)

The app supports both: 
- single-comment prediction
- batch prediction on scraped product reviews (with navigation)

## Tech Stack 
* Python<br>
* TensorFlow / Keras<br>
* HuggingFace Transformers<br>
* BeautifulSoup<br>
* CustomTkinter<br>

## Models 
### 1)Custom Model: Keras LSTM (3 classes)

Training pipeline:<br>
Load yorumlar.csv<br>
Basic text cleanup<br>
Label encoding<br>
Tokenizer (num_words=5000)<br>
Padding<br>
Model:<br>
* Embedding(5000, 32)
* LSTM(64, dropout=0.2, recurrent_dropout=0.2)
* Dense(3, softmax)

Outputs: Negative / Positive / Neutral

Saved artifacts:<br>
* sentiment_model.h5
* tokenizer.pkl

### 2) Transformer: BERT (Pretrained)

Using HuggingFace:
* pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

This returns label + score.<br>
We apply a simple but effective confidence threshold: score < 0.75 ⇒ Neutral

This helps avoid overconfident incorrect predictions. 

<img width="501" height="510" alt="Ekran Resmi 2026-02-26 17 34 10" src="https://github.com/user-attachments/assets/713fa420-ad56-49d5-a34a-18fa59884e96" />
<img width="501" height="521" alt="Ekran Resmi 2026-02-26 17 35 04" src="https://github.com/user-attachments/assets/68f25b49-07d0-41db-b14d-d89a3038231c" />

