import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import joblib

# Veriyi yükle
df = pd.read_csv('yorumlar.csv', delimiter=';').iloc[1:]
df['Metin'] = df['Metin'].str.replace(r'^\w+\s', '', regex=True)

# Veriyi hazırla
X = df['Metin']
y = LabelEncoder().fit_transform(df['Durum'].astype(int))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Metin verilerini vektörlere dönüştürme
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
max_sequence_length = max([len(x) for x in X_train_sequences])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Modeli oluştur
model = Sequential()
model.add(Embedding(5000, 32, input_length=max_sequence_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=10, batch_size=64)

# Eğitilmiş modeli diske kaydet
model.save('sentiment_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
