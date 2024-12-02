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
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema(enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)

milvus_client.create_collection(collection_name, schema=schema, consistency_level="Strong")
print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))


rng = np.random.default_rng(seed=19530)

# for j in range(10):
#     rows = [{"id": j*10000 + i, "embeddings": rng.random((1, dim))[0]} for i in range(10000)]
#     print(fmt.format("Start inserting entities"))
#     insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
#     print(fmt.format("Inserting entities done"))
#     print(insert_result)

for j in range(1):
    rows = [{"id": j*100 + i, "embeddings": rng.random((1, dim))[0]} for i in range(10000)]
    print(fmt.format("Start inserting entities"))
    insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
    print(fmt.format("Inserting entities done"))
    print(insert_result)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", metric_type="L2")

print(fmt.format("Start create index"))
milvus_client.create_index(collection_name, index_params)

index_names = milvus_client.list_indexes(collection_name)
print(f"index names for {collection_name}:", index_names)
for index_name in index_names:
    index_info = milvus_client.describe_index(collection_name, index_name=index_name)
    print(f"index info for index {index_name} is:", index_info)

print(fmt.format("Start load collection"))
milvus_client.load_collection(collection_name)

vectors_to_search = rng.random((1, dim))
print(fmt.format(f"Start search with retrieve serveral fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["id"])
for hits in result:
    for hit in hits:

        print(f"hit: {hit}")