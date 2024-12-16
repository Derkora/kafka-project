# Kafka project

## Arsitektur
![arsi](img/image.png)

### Tahap 0
run docker
```bash
docker-compose up -d
```

download dataset
```bash
bash dataset/download.sh
```

kafka
```bash
docker exec -it kafka kafka-topics.sh --create --topic samsung-stock --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics.sh --create --topic stock-input --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### login minio
```txt
minio/minio123
```

### Tahap 1 (preset)
```bash
cd preset
python3 producer.py
python3 consumer.py
python3 train.py
```
atau bash start.sh dan liat pada minio untuk memastikan apakah sudah selesai

### Tahap 2 (main)
```bash
cd main
python3 api/app.py 
python3 kafka/consumer.py
```

web
```bash
cd main/web
python3 -m http.server 8080
```