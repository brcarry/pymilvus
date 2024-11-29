import asyncio
import time
from pymilvus import DataType
from pymilvus import MilvusClient
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient
import numpy as np

fmt = "\n=== {:30} ===\n"
dim = 768
collection_name = "hello_milvus"
URI = "http://localhost:19530"
# URI = "example.db"


rng = np.random.default_rng(seed=19530)

milvus_client = MilvusClient(uri = URI)

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
NUM = 1000
for j in range(100):
    rows = [{"id": j*NUM + i, "embeddings": rng.random((1, dim))[0]} for i in range(NUM)]
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




# 同步查询函数
def sync_search(num_queries):
    vectors_to_search = rng.random((1, dim))
    start_time = time.time()
    for _ in range(num_queries):
        milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["id"])
    end_time = time.time()
    return end_time - start_time

async_client = AsyncMilvusClient(uri=URI)

# 异步查询函数
async def async_search(num_queries):
    # async_client = AsyncMilvusClient(uri="example.db")
    vectors_to_search = rng.random((1, dim))
    start_time = time.time()
    tasks = []
    for _ in range(num_queries):
        task = async_client.async_search(collection_name, vectors_to_search, limit=3, output_fields=["id"])
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    return end_time - start_time

import asyncio

# 执行查询并打印时间
query_counts = [10, 100, 1000, 10000, 20000, 30000, 40000, 50000, 1000000]
loop = asyncio.get_event_loop()
# query_counts = [10, 100, 1000]
for count in query_counts:
    sync_time = sync_search(count)
    print(f"Sync search for {count} queries took {sync_time:.2f} seconds")

    # 异步查询需要在asyncio的事件循环中运行
    async def measure_async_search():
        async_time = await async_search(count)
        print(f"Async search for {count} queries took {async_time:.2f} seconds")
        return async_time

    loop.run_until_complete(measure_async_search())