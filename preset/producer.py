from confluent_kafka import Producer
import pandas as pd
import json
import os

# Konfigurasi Kafka
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "samsung-stock"

# Konfigurasi Kafka Producer
conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'samsung-stock-producer'
}

producer = Producer(conf)

print("Kafka Producer siap...")

# Fungsi untuk melaporkan hasil pengiriman pesan
def delivery_report(err, msg):
    if err:
        print(f"Pengiriman pesan gagal: {err}")
    else:
        print(f"Pesan terkirim ke topic {msg.topic()} [partisi {msg.partition()}] @ offset {msg.offset()}")

# Path ke file dataset
DATASET_PATH = "samsung_stock.csv"

# Memeriksa keberadaan file
if not os.path.exists(DATASET_PATH):
    print(f"File {DATASET_PATH} tidak ditemukan. Pastikan file tersedia.")
    exit(1)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset berhasil dimuat. Total {len(df)} baris data.")
except Exception as e:
    print(f"Gagal membaca dataset: {e}")
    exit(1)

# Iterasi dataset dan kirim pesan ke Kafka
for index, row in df.iterrows():
    try:
        # Pastikan kolom date valid
        if pd.notna(row['Date']):
            message = {
                "date": row["Date"],
                "close": row["Close"]
            }
            
            # Kunci pesan (berdasarkan tanggal)
            key = row["Date"].replace("-", "")
            
            # Kirim pesan ke Kafka
            producer.produce(
                topic=TOPIC_NAME,
                key=key,
                value=json.dumps(message),
                callback=delivery_report
            )
            
            producer.poll(0)  # Memberi waktu untuk callback
            
        else:
            print(f"Data kosong pada indeks {index}, dilewati.")
    
    except Exception as e:
        print(f"Error saat mengirim data indeks {index}: {e}")

# Tunggu semua pesan selesai dikirim
producer.flush()
print("Semua pesan telah berhasil dikirim ke Kafka.")
