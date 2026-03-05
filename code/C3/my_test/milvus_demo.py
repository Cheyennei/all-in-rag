import os
import dashscope

from merge_data import JsonProcess
from img_process import ImageProcess
from milvus_client import GetMilvus

if __name__ == "__main__":
    vector_save = GetMilvus()
    vector_save.create_collection()

    img = ImageProcess()
    img_embedding = img.batch_img_to_vec(r"G:\Python_file\all-in-rag\code\C3\my_test\data")

    merge = JsonProcess()
    merged_list = merge.merge_json_dict(img_embedding)

    vector_save.insert_collection(merged_list)
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
    # text_query = "悬崖上的巨龙"
    # search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
    # text_resp = dashscope.TextEmbedding.call(
    #     model="text-embedding-v4",
    #     input=text_query,
    #     api_key=os.getenv("dashscope_api_key"),
    #     dimension=1024,
    #     output_type="dense&sparse"
    # )
    # print(text_resp)
    # dense_embedding = text_resp.output['embeddings'][0]['embedding']
    # sparse_embedding = text_resp.output['embeddings'][0]['sparse_embedding']
    # res = client.search(
    #     collection_name=COLLECTION_NAME,
    #     anns_field="dense_vector",
    #     data=[dense_embedding],
    #     search_params={"metric_type": "COSINE"},
    #     limit=5,
    #     filter=search_filter,
    # )
    # for hits in res:
    #     for hit in hits:
    #         print(hit)

    # json_path = r"G:\Python_file\all-in-rag\data\C4\metadata\dragon.json"
    # resp = json_text_to_vec(json_path)
    # img_path = r"G:\Python_file\all-in-rag\code\C3\my_test\data"
    # img_vec = batch_img_to_vec(img_path)
    # res = merge_json_dict(r"G:\Python_file\all-in-rag\data\C4\metadata\dragon.json",img_vec,resp)
    # print(res)
    # create_collection(COLLECTION_NAME)
    # insert_collection(COLLECTION_NAME, res)


