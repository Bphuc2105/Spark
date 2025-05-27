from kafka import KafkaConsumer
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaToElasticsearch:
    def __init__(self, es_host="http://localhost:9200", batch_size=100):
        self.es = Elasticsearch(es_host)
        self.batch_size = batch_size
        self.batch_docs = []
        
    def create_index_with_mapping(self, index_name, reset_index=False):
        """Create Elasticsearch index with appropriate mappings"""
        if reset_index and self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted.")

        if not self.es.indices.exists(index=index_name):
            # Create mapping based on expected data structure
            if index_name == 'articles':
                mapping = {
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "link": {"type": "keyword"},  # For duplicate checking
                            "date": {
                                "type": "date",
                                "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"
                            },
                            "source": {"type": "keyword"}
                        }
                    }
                }
            elif index_name == 'prices':
                mapping = {
                    "mappings": {
                        "properties": {
                            "symbol": {"type": "keyword"},
                            "price": {"type": "float"},
                            "volume": {"type": "long"},
                            "date": {
                                "type": "date",
                                "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"
                            }
                        }
                    }
                }
            else:
                # Generic mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "date": {
                                "type": "date",
                                "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"
                            }
                        }
                    }
                }
            
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"Index '{index_name}' created with custom mapping.")
        else:
            logger.info(f"Index '{index_name}' already exists.")

    def check_duplicate(self, doc, index_name):
        """Check if document already exists based on unique fields"""
        try:
            if index_name == 'articles':
                # Check duplicate based on link
                if 'link' not in doc:
                    return False  # No link field, cannot check for duplicate
                
                query = {
                    "query": {
                        "term": {
                            "link.keyword": doc['link']
                        }
                    }
                }
                response = self.es.search(index=index_name, body=query, size=1)
                return response['hits']['total']['value'] > 0
                
            elif index_name == 'prices':
                # Check duplicate based on date + symbol
                if 'date' not in doc or 'symbol' not in doc:
                    return False  # Missing required fields, cannot check for duplicate
                
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"symbol.keyword": doc['symbol']}},
                                {"term": {"date": doc['date']}}
                            ]
                        }
                    }
                }
                response = self.es.search(index=index_name, body=query, size=1)
                return response['hits']['total']['value'] > 0
                
        except Exception as e:
            logger.error(f"Error checking duplicate for {index_name}: {e}")
            return False  # If error occurs, proceed with indexing
        
        return False

    def add_to_batch(self, doc, index_name):
        """Add document to batch for bulk indexing"""
        # Check for duplicates
        if self.check_duplicate(doc, index_name):
            logger.info(f"Duplicate document found in {index_name}, skipping...")
            return
        
        es_doc = {
            "_index": index_name,
            "_source": doc
        }
        
        self.batch_docs.append(es_doc)
        
        # If batch is full, index it
        if len(self.batch_docs) >= self.batch_size:
            self.flush_batch()

    def flush_batch(self):
        """Index the current batch of documents"""
        if not self.batch_docs:
            return
            
        try:
            success, failed = bulk(self.es, self.batch_docs, raise_on_error=False)
            logger.info(f"Indexed {success} documents successfully.")
            
            if failed:
                logger.warning(f"{len(failed)} documents failed to index:")
                for error in failed:
                    logger.error(error)
                    
        except BulkIndexError as e:
            logger.error("Bulk index error occurred:")
            for item in e.errors:
                logger.error(item)
        finally:
            # Clear the batch
            self.batch_docs = []

    def consume_and_index(self, topic_to_index_mapping, bootstrap_servers=['localhost:9093']):
        """Consume from Kafka topics and index to Elasticsearch"""
        
        # Create consumers for each topic
        consumers = {}
        for topic, index_name in topic_to_index_mapping.items():
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',  # Start from latest messages
                enable_auto_commit=True,
                group_id=f'{topic}_to_es_group'
            )
            consumers[topic] = consumer
            
            # Create index for this topic
            self.create_index_with_mapping(index_name, reset_index=False)
        
        logger.info(f"Starting to consume from topics: {list(topic_to_index_mapping.keys())}")
        
        try:
            # Poll messages from all consumers
            for topic, consumer in consumers.items():
                index_name = topic_to_index_mapping[topic]
                for message in consumer:
                    try:
                        data = message.value
                        logger.info(f"Received message from {topic}: {data}")
                        
                        # Add to batch for indexing
                        self.add_to_batch(data, index_name)
                        
                    except Exception as e:
                        logger.error(f"Error processing message from {topic}: {e}")
                        continue
                        
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            # Flush any remaining documents
            self.flush_batch()
            
            # Close consumers
            for consumer in consumers.values():
                consumer.close()

def main():
    # Initialize the Kafka to Elasticsearch processor
    processor = KafkaToElasticsearch(
        es_host="http://localhost:9200",
        batch_size=50  # Adjust batch size as needed
    )
    
    # Define topics to consume from
    topics = ['news_articles', 'new_prices']
    
    # Start consuming and indexing
    processor.consume_and_index(topics)

if __name__ == "__main__":
    main()

# Alternative: Simple single-topic consumer (fixed version of your original code)
def simple_consumer_example():
    """Simple example for consuming from a single topic"""
    # Initialize Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    
    # Create consumer for news articles
    consumer = KafkaConsumer(
        'news_articles',
        bootstrap_servers=['localhost:9093'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    # Create index if it doesn't exist
    index_name = 'news_articles'
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "date": {"type": "date"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
    
    try:
        for message in consumer:
            article = message.value
            print(f"Received article: {article.get('title', 'No title')}")
            
            # Index to Elasticsearch
            try:
                response = es.index(index=index_name, body=article)
                print(f"Indexed article with ID: {response['_id']}")
            except Exception as e:
                print(f"Error indexing article: {e}")
                
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        consumer.close()