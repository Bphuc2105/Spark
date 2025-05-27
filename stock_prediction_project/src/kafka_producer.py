# data/kafka_dual_producer.py

import time
import json
import csv
import os
import subprocess
from kafka import KafkaProducer

# Config
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9093")
ARTICLE_TOPIC = os.environ.get("NEWS_ARTICLES_TOPIC", "articles")
PRICES_TOPIC = os.environ.get("STOCK_PRICES_TOPIC", "prices")

ARTICLE_CSV = "data/articles.csv"
PRICE_CSV = "data/prices.csv"

def create_kafka_topic(topic_name, partitions=1, replication=1):
    """Create a Kafka topic using shell command."""
    try:
        subprocess.run([
            "kafka-topics.sh",
            "--create",
            "--topic", topic_name,
            "--bootstrap-server", KAFKA_BROKER,
            "--partitions", str(partitions),
            "--replication-factor", str(replication)
        ], check=True)
        print(f"✅ Created topic: {topic_name}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Topic creation failed or already exists: {topic_name}")
    except FileNotFoundError:
        print("⚠️ kafka-topics.sh not found — skipping manual topic creation.")

def send_articles_to_kafka():
    """Send articles CSV to Kafka topic."""
    print(f"📤 Sending articles from {ARTICLE_CSV} to topic '{ARTICLE_TOPIC}'")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

        with open(ARTICLE_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = 0
            
            for row in reader:
                # Check required fields for articles
                if not all(field in row and row[field] for field in ["date", "text"]):
                    print(f"⚠️ Missing required fields (date, text) in row: {row}")
                    continue

                # Generate ID if missing
                if 'id' not in row or not row['id']:
                    row['id'] = f"article_{int(time.time()*1000)}_{row_count}"

                # Clean up the row data
                article_data = {
                    'id': row['id'],
                    'date': row['date'],
                    'link': row.get('link', ''),
                    'text': row['text'],
                    'title': row.get('title', '')
                }

                # Use article ID as key for better partitioning
                key = row['id'].encode('utf-8')
                producer.send(ARTICLE_TOPIC, key=key, value=article_data)
                print(f"  ✅ Sent article: {row['id']} to {ARTICLE_TOPIC}")
                row_count += 1

        producer.flush()
        producer.close()
        print(f"✅ Finished sending {row_count} articles to topic {ARTICLE_TOPIC}")

    except Exception as e:
        print(f"❌ Articles Kafka error: {e}")
        import traceback
        traceback.print_exc()

def send_prices_to_kafka():
    """Send stock prices CSV to Kafka topic."""
    print(f"📤 Sending prices from {PRICE_CSV} to topic '{PRICES_TOPIC}'")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

        with open(PRICE_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = 0
            
            for row in reader:
                # Check required fields for prices
                if not all(field in row and row[field] for field in ["date", "symbol"]):
                    print(f"⚠️ Missing required fields (date, symbol) in row: {row}")
                    continue

                # Generate ID if missing
                if 'id' not in row or not row['id']:
                    row['id'] = f"price_{row.get('symbol', 'unknown')}_{int(time.time()*1000)}_{row_count}"

                # Convert numeric fields with error handling
                try:
                    close_price = float(row.get('close', 0)) if row.get('close') else None
                    open_price = float(row.get('open', 0)) if row.get('open') else None
                    volume = int(row.get('volume', 0)) if row.get('volume') else None
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid numeric data in row {row['id']}: {e}")
                    continue

                # Clean up the row data
                price_data = {
                    'id': row['id'],
                    'date': row['date'],
                    'close': close_price,
                    'volume': volume,
                    'source': row.get('source', ''),
                    'symbol': row['symbol'],
                    'open': open_price
                }

                # Use symbol as key for better partitioning (stocks with same symbol go to same partition)
                key = row['symbol'].encode('utf-8')
                producer.send(PRICES_TOPIC, key=key, value=price_data)
                print(f"  ✅ Sent price: {row['symbol']} ({row['id']}) to {PRICES_TOPIC}")
                row_count += 1

        producer.flush()
        producer.close()
        print(f"✅ Finished sending {row_count} price records to topic {PRICES_TOPIC}")

    except Exception as e:
        print(f"❌ Prices Kafka error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Step 1: Create topics
    print("🔧 Creating Kafka topics...")
    create_kafka_topic(ARTICLE_TOPIC)
    create_kafka_topic(PRICES_TOPIC)

    # Step 2: Send articles to topic
    print("\n📰 Processing articles...")
    send_articles_to_kafka()

    # Step 3: Send prices to topic
    print("\n📈 Processing stock prices...")
    send_prices_to_kafka()
    
    print("\n🎉 All data sent successfully!")
