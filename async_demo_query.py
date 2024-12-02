import time
import numpy as np
from pymilvus import DataType
from pymilvus import Milvus
from pymilvus import MilvusClient
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient

fmt = "\n=== {:30} ===\n"
# dim = 8
dim = 768
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
print("has_collection:", has_collection)

rng = np.random.default_rng(seed=19530)

print(fmt.format("Start load collection"))
milvus_client.load_collection(collection_name)

vectors_to_search = rng.random((1, dim))
print(fmt.format(f"Start search with retrieve serveral fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["id"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")