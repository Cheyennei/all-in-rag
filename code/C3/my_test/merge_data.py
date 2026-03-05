import json
import os

import dashscope



class JsonProcess:
    JSON_PATH = r"G:\Python_file\all-in-rag\data\C4\metadata\dragon.json"
    def __init__(self, json_path:str = JSON_PATH):
        self.json_path = json_path
        self.all_embedding = None

    # 返回dense和sparse向量
    def json_text_to_vec(self):
        with open(self.json_path, "r",encoding='utf-8') as f:
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
        self.all_embedding = resp.output['embeddings']
        return self.all_embedding



    def merge_json_dict(self,img_embedding):
        """
        :param json_path:
        :param data_dict:使用json_text_to_vec后返回的数据
        """
        self.all_embedding = self.json_text_to_vec()
        with open(self.json_path, "r",encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list) or not isinstance(img_embedding, list):
            raise TypeError("输入必须是两个列表（每个元素是字典）")

        merged_list = []
        min_len = min(len(dataset), len(img_embedding))

        # 遍历每个索引，合并单个字典
        for idx in range(min_len):
            sparse_dict = {"sparse_vector":{item['index']: item['value'] for item in self.all_embedding[idx]['sparse_embedding']}}
            dense_dict = {"dense_vector":self.all_embedding[idx]['embedding']}
            print(f"{'='*50}")
            print(sparse_dict)
            print(f"{'=' * 50}")
            print(f"{'*' * 50}")
            print(dense_dict)
            print(f"{'*' * 50}")
            # 取出当前索引的单个字典
            dict1 = dataset[idx]
            dict2 = img_embedding[idx]

            merged_dict = {**dict1, **dict2}
            merged_vec = {**sparse_dict, **dense_dict}
            merged_dict = {**merged_dict,**merged_vec}
            merged_list.append(merged_dict)


        return merged_list