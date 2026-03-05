from typing import List

from pymilvus import MilvusClient, DataType


class GetMilvus:
    """
    实例化可以拿到一个Milvus客户端
    """

    URI="http://localhost:19530"
    COLLECTION_NAME = "image_collection"

    def __init__(self, uri: str = URI, collection_name: str = COLLECTION_NAME):
        self.uri = uri
        self.collection_name = collection_name
        self.client = MilvusClient(self.uri)

    def create_collection(self):
        if self.collection_name in self.client.list_collections():
            print(f"Collection '{self.collection_name}' already exists, skip create.")
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="img_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="img_id", datatype=DataType.VARCHAR, max_length=100),
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=256),
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256),
        schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=4096),
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64),
        schema.add_field(field_name="location", datatype=DataType.VARCHAR, max_length=128),
        schema.add_field(field_name="environment", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR),
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="my_id", index_type="AUTOINDEX")
        # 必须与 schema 中的向量字段同名: img_vector
        index_params.add_index(
            field_name="img_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )


    def insert_collection(self, data: List[dict]):
        if not data:
            print("没有可插入的数据")
            return None
        return self.client.insert(collection_name=self.collection_name, data=data)

