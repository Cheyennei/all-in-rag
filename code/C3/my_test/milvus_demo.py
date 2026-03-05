import os
import base64
from pathlib import Path
from typing import List
import dashscope
import json
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

load_dotenv()

client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "image_collection"


def img_to_base64(img_path: str) -> str:
    ext = Path(img_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(img_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{base64_image}"


def img_to_vec(img_data: str):
    payload = [{"image": img_data}]
    resp = dashscope.MultiModalEmbedding.call(
        model="tongyi-embedding-vision-flash",
        input=payload,
        api_key=os.getenv("dashscope_api_key"),
    )
    return resp.output["embeddings"][0]["embedding"]


def batch_img_to_vec(input_dir: str):
    data = []
    supported_formats = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"目录不存在: {input_dir}")
        return []

    img_list = [f for f in input_path.glob("*") if f.suffix.lower() in supported_formats]
    total = len(img_list)
    print(f"找到 {total} 张图片，开始批量处理...")

    for idx, img_file in enumerate(img_list, 1):
        img_path = str(img_file.absolute())
        img_name = img_file.name
        print(f"[{idx}/{total}] 处理: {img_name}")
        try:
            base_str = img_to_base64(img_path)
            img_vec = img_to_vec(base_str)
            data.append({"my_id":idx,"name": img_name, "img_vector": img_vec})
        except Exception as e:
            print(f"处理失败: {img_name}, error={e}")

    print(f"处理完成，成功 {len(data)} 张")
    return data

def json_text_to_vec(json_path:str):
    with open(json_path, "r",encoding='utf-8') as f:
        dataset = json.load(f)
    docs = []
    for item in dataset:
        temp = str(item.get("title","")+item.get("description","")+
                    item.get("category","")+item.get("location","")
                   )

        docs.append(temp)
    # print(f"{'='*50}\n{docs}\n{'='*50}")
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=docs,
        api_key=os.getenv("dashscope_api_key"),
        dimension = 1024,
        output_type="dense&sparse"
    )
    return resp.output['embeddings']

def create_collection(collection_name: str):
    if collection_name in client.list_collections():
        print(f"Collection '{collection_name}' already exists, skip create.")
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

    index_params = client.prepare_index_params()
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

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def insert_collection(collection_name: str, data: List[dict]):
    if not data:
        print("没有可插入的数据")
        return None
    return client.insert(collection_name=collection_name, data=data)

def merge_json_dict(json_path,data_dict,resp):


    with open(json_path, "r",encoding='utf-8') as f:
        dataset = json.load(f)
    if not isinstance(dataset, list) or not isinstance(data_dict, list):
        raise TypeError("输入必须是两个列表（每个元素是字典）")

    merged_list = []
    min_len = min(len(dataset), len(data_dict))

    # 遍历每个索引，合并单个字典
    for idx in range(min_len):
        sparse_dict = {"sparse_vector":{item['index']: item['value'] for item in resp[idx]['sparse_embedding']}}
        dense_dict = {"dense_vector":resp[idx]['embedding']}
        print(f"{'='*50}")
        print(sparse_dict)
        print(f"{'=' * 50}")
        print(f"{'*' * 50}")
        print(dense_dict)
        print(f"{'*' * 50}")
        # 取出当前索引的单个字典
        dict1 = dataset[idx]
        dict2 = data_dict[idx]

        merged_dict = {**dict1, **dict2}
        merged_vec = {**sparse_dict, **dense_dict}
        merged_dict = {**merged_dict,**merged_vec}
        merged_list.append(merged_dict)


    return merged_list

if __name__ == "__main__":
    # img_query = "不是一条龙"
    # img_resp = dashscope.MultiModalEmbedding.call(
    #     model="tongyi-embedding-vision-flash",
    #     input=[{"text": img_query}],
    #     api_key=os.getenv("dashscope_api_key"),
    # )
    # embedding = img_resp.output["embeddings"][0]["embedding"]
    # res = client.search(
    #     collection_name=COLLECTION_NAME,
    #     anns_field="img_vector",
    #     data=[embedding],
    #     search_params={"metric_type": "COSINE"},
    #     limit=3,
    # )
    text_query = "悬崖上的巨龙"
    search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
    text_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=text_query,
        api_key=os.getenv("dashscope_api_key"),
        dimension=1024,
        output_type="dense&sparse"
    )
    print(text_resp)
    dense_embedding = text_resp.output['embeddings'][0]['embedding']
    sparse_embedding = text_resp.output['embeddings'][0]['sparse_embedding']
    res = client.search(
        collection_name=COLLECTION_NAME,
        anns_field="dense_vector",
        data=[dense_embedding],
        search_params={"metric_type": "COSINE"},
        limit=5,
        filter=search_filter,
    )
    for hits in res:
        for hit in hits:
            print(hit)

    # json_path = r"G:\Python_file\all-in-rag\data\C4\metadata\dragon.json"
    # resp = json_text_to_vec(json_path)
    # img_path = r"G:\Python_file\all-in-rag\code\C3\my_test\data"
    # img_vec = batch_img_to_vec(img_path)
    # res = merge_json_dict(r"G:\Python_file\all-in-rag\data\C4\metadata\dragon.json",img_vec,resp)
    # print(res)
    # create_collection(COLLECTION_NAME)
    # insert_collection(COLLECTION_NAME, res)


