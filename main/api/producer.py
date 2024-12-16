from confluent_kafka import Producer
import json

KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "stock-input"

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def delivery_report(err, msg):
    if err:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")

def send_to_kafka(dates, closes):
    if len(dates) != len(closes):
        raise ValueError("Dates and closes must have the same length")

    message = {
        "dates": dates,
        "closes": closes
    }
    producer.produce(TOPIC_NAME, json.dumps(message), callback=delivery_report)
    producer.flush()
