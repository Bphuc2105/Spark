import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError

def create_index_with_date_mapping(es, index_name, df, reset_index=False):
    if reset_index and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted.")

    if not es.indices.exists(index=index_name):
        
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
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created with custom date mapping.")
    else:
        print(f"Index '{index_name}' already exists.")

def dataframe_to_es_docs(df, index_name):
    for i, row in df.iterrows():
        doc = row.to_dict()
        yield {
            "_index": index_name,
            "_id": i,
            "_source": doc
        }

def upload_csv_to_elasticsearch(csv_path, index_name, es_host="http://localhost:9200", reset_index=False):
    es = Elasticsearch(es_host)
    df = pd.read_csv(csv_path)

    
    

    create_index_with_date_mapping(es, index_name, df, reset_index=reset_index)

    try:
        success, failed = bulk(es, dataframe_to_es_docs(df, index_name), raise_on_error=False)
        print(f"Uploaded {success} documents to '{index_name}'.")
        if failed:
            print("Some documents failed to index:")
            for error in failed:
                print(error)
    except BulkIndexError as e:
        print("Bulk index error occurred:")
        for item in e.errors:
            print(item)


upload_csv_to_elasticsearch("data/prices.csv", "prices", reset_index=True)
