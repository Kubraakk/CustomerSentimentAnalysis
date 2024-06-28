import tkinter as tk
import customtkinter
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import requests
from bs4 import BeautifulSoup

header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/80.0.3987.163 Safari/537.36 "
}

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Yorum Duygu Tahmini')
        self.root.geometry("600x550")  # Pencere boyutunu ayarla

        self.all_predictions = []  # Tahminler tuttulan liste
        self.current_index = -1  # Tahminin mevcut dizini

        # İleri ve geri butonları
        self.prev_button = customtkinter.CTkButton(master=self.root, text="Önceki",
                                                   command=self.show_previous_prediction)
        self.prev_button.grid(row=7, column=0, padx=10, pady=10, sticky='w')
        self.next_button = customtkinter.CTkButton(master=self.root, text="Sonraki", command=self.show_next_prediction)
        self.next_button.grid(row=7, column=0, padx=10, pady=10, sticky='e')


        self.label = tk.Label(root, text='Ürün Linki:', font=("Arial", 12))
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='n')

        self.link_entry = customtkinter.CTkEntry(master=self.root, placeholder_text="Ürün Linki", width=500, height=50)
        self.link_entry.grid(row=1, column=0, padx=10, pady=10, sticky='n')

        self.scrape_button = customtkinter.CTkButton(master=self.root, text="Ürün Yorumlarını Analiz Et", command=self.scrape_and_predict)
        self.scrape_button.grid(row=2, column=0, padx=10, pady=10, sticky='n')

        self.label = tk.Label(root, text='Yorum:', font=("Arial", 12))
        self.label.grid(row=3, column=0, padx=10, pady=10, sticky='n')

        self.text_entry = customtkinter.CTkEntry(master=self.root, placeholder_text="Yorum", width=500, height=50)
        self.text_entry.grid(row=4, column=0, padx=10, pady=10, sticky='n')

        self.predict_button = customtkinter.CTkButton(master=self.root, text="Yorumu Analiz Et", command=self.predict_with_entry_text)
        self.predict_button.grid(row=5, column=0, padx=10, pady=10, sticky='n')

        self.result_label = tk.Label(root, text='', font=("Arial", 12), foreground="blue", justify='left', wraplength=500)
        self.result_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='n')

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Eğitilmiş modeli ve tokenizer'i yükle
        self.keras_model = load_model('sentiment_model.h5')
        self.tokenizer = joblib.load('tokenizer.pkl')

        # Transformer modelini yükle
        self.transformer_classifier = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

    def predict_with_entry_text(self):
        text = self.text_entry.get() # Giriş kutusundan metni al
        if text.strip():
                # Keras modeliyle tahmin yap
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=125)  # Max sequence length'ini uygun şekilde değiştirin
            keras_prediction = self.keras_model.predict(padded_sequence)[0]
            keras_sentiment_label = ['Olumsuz', 'Olumlu', 'Nötr']
            keras_result_index = keras_prediction.argmax()
            keras_result_label = keras_sentiment_label[keras_result_index]

            # Transformer modeliyle tahmin yap
            transformer_result = self.transformer_classifier(text)[0]
            transformer_sentiment_label = transformer_result['label']
            transformer_confidence = transformer_result['score']

            threshold_value = 0.75

                # Yeniden etiketleme
            if transformer_confidence < threshold_value:
                transformer_sentiment_label = "Nötr"
            else:
                if transformer_sentiment_label == "positive":
                    transformer_sentiment_label = "Olumlu"
                elif transformer_sentiment_label == "negative":
                    transformer_sentiment_label = "Olumsuz"
                else:
                    transformer_sentiment_label = transformer_result['label']

            prediction_text = f'Yorum: {text}\n\nEğitilmiş Model Tahmini: {keras_result_label} - Skor: {max(keras_prediction):.2f}\n\nTransformer Model Tahmini: {transformer_sentiment_label}- Skor: {transformer_confidence:.2f}\n\n'

            self.text_entry.delete(0, tk.END)  # Yorum kutusunu temizle
                #self.text_entry.insert(0, text)
            self.link_entry.delete(0, tk.END)  # Link kutusunu temizle
                #self.link_entry.insert(0, text)


        self.result_label.configure(text=prediction_text, anchor='center')





    def predict_sentiments(self, reviews):
        #texts = self.text_entry.get().split('\n')
        self.all_predictions = []
        self.current_index = -1

        for text in reviews:
            if text.strip():
                # Keras modeliyle tahmin yap
                sequence = self.tokenizer.texts_to_sequences([text])
                padded_sequence = pad_sequences(sequence, maxlen=125)  # Max sequence length'ini uygun şekilde değiştirin
                keras_prediction = self.keras_model.predict(padded_sequence)[0]
                keras_sentiment_label = ['Olumsuz', 'Olumlu', 'Nötr']
                keras_result_index = keras_prediction.argmax()
                keras_result_label = keras_sentiment_label[keras_result_index]

                # Transformer modeliyle tahmin yap
                transformer_result = self.transformer_classifier(text)[0]
                transformer_sentiment_label = transformer_result['label']
                transformer_confidence = transformer_result['score']

                threshold_value = 0.75

                # Yeniden etiketleme
                if transformer_confidence < threshold_value:
                    transformer_sentiment_label = "Nötr"
                else:
                    if transformer_sentiment_label == "positive":
                        transformer_sentiment_label = "Olumlu"
                    elif transformer_sentiment_label == "negative":
                        transformer_sentiment_label = "Olumsuz"
                    else:
                        transformer_sentiment_label = transformer_result['label']

                prediction_text = (f'Yorum: {text}\n\nEğitilmiş Model Tahmini: {keras_result_label} - Skor: {max(keras_prediction):.2f}\n\n'
                                   f'Transformer Model Tahmini: {transformer_sentiment_label}- Skor: {transformer_confidence:.2f}\n\n')
                self.all_predictions.append(prediction_text)
                self.text_entry.delete(0, tk.END)  # Yorum kutusunu temizle
                self.link_entry.delete(0, tk.END)  # Link kutusunu temizle

        result_text = '\n'.join(self.all_predictions)
        self.result_label.configure(text=result_text, anchor='center')

        # Varsa ilk tahmini göster
        if self.all_predictions:
            self.show_next_prediction()

    def show_previous_prediction(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.result_label.configure(text=self.all_predictions[self.current_index])

    def show_next_prediction(self):
        if self.current_index < len(self.all_predictions) - 1:
            self.current_index += 1
            self.result_label.configure(text=self.all_predictions[self.current_index])



    def scrape_and_predict(self):
        product_url = self.link_entry.get()
        if product_url:
            reviews = self.scrape_reviews(product_url)
            if reviews:
                #text = '\n'.join(reviews)
                #self.text_entry.insert(0, text)
                self.predict_sentiments(reviews)

                self.text_entry.delete(0, tk.END)  # Yorum kutusunu temizle

                self.link_entry.delete(0, tk.END)  # Link kutusunu temizle


    def scrape_reviews(self, product_url):
        request = requests.get(product_url, headers=header)
        tumVeriler = BeautifulSoup(request.content, "html.parser")

        # Tüm comment_div öğelerini bul
        comment_divs = tumVeriler.find_all("div", {"class": "a-section review-views celwidget"})
        reviews = []

        for comment_div in comment_divs:
            # Her comment_div içindeki tüm span öğelerini bul
            spans = comment_div.find_all("span", {"class": "a-size-base review-text"})
            for span in spans:
                # "Daha fazla bilgi" başlıklı bölümleri sil
                reviews.append(span.text.strip().replace("Daha fazla bilgi", ""))
        return reviews

if __name__ == '__main__':
    root = customtkinter.CTk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()

