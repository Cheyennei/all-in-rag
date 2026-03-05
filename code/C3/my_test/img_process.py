import os
from dotenv import load_dotenv
from pathlib import Path
import base64
from typing import Optional, List,Dict
import dashscope
load_dotenv()

class ImageProcess:
    """
    将图片文件转换为base64格式，方便传给大模型
    """
    SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
    def __init__(self,api_key:Optional[str]=None):

        self.api_key = api_key or os.getenv("dashscope_api_key")
        if not self.api_key:
            raise ValueError("请配置dashscope_api_key环境变量，或传入api_key参数")

        self.img_data = None  # base64编码后的图片数据
        self.img_vector = None  # 最终生成的图片向量

    def _validate_img_path(self,img_path:str)->None:
        if not img_path:
            raise ValueError("图片路径不能为空")
        img_path_obj = Path(img_path)
        # 检查文件是否存在
        if not img_path_obj.exists():
            raise FileNotFoundError(f"图片文件不存在：{img_path}")
        # 检查是否是文件（而非文件夹）
        if not img_path_obj.is_file():
            raise IsADirectoryError(f"路径不是文件：{img_path}")
        # 检查是否是支持的图片格式
        supported_formats = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ext = img_path_obj.suffix.lower()
        if ext not in supported_formats:
            raise ValueError(f"不支持的图片格式：{ext}，支持格式：{supported_formats}")

    def img_to_base64(self,img_path: str) -> str:
        ext = Path(img_path).suffix.lower().lstrip(".")
        ext = "jpeg" if ext == "jpg" else ext

        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            img_data = f"data:image/{ext};base64,{base64_image}"

        return img_data

    def img_to_vec(self,img_path: str):
        try:
            base64_str = self.img_to_base64(img_path)
            payload = [{"image": base64_str}]
            resp = dashscope.MultiModalEmbedding.call(
                model=self.EMBEDDING_MODEL,
                input=payload,
                api_key=self.api_key,
            )
            # 校验API返回结果
            if not resp.output or "embeddings" not in resp.output:
                raise ValueError(f"API返回格式异常：{resp}")

            self.img_vector = resp.output["embeddings"][0]["embedding"]
            return self.img_vector
        except Exception as e:
            raise RuntimeError(f"图片转向量失败：{str(e)}") from e

    def batch_img_to_vec(self, input_dir: str)-> List[Dict]:
        self.batch_success_count = 0
        self.batch_failed_count = 0
        self.batch_failed_files = []
        data = []
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"目录不存在: {input_dir}")
            return []

        img_list = [f for f in input_path.glob("*") if f.suffix.lower() in self.SUPPORTED_FORMATS]
        total = len(img_list)
        print(f"找到 {total} 张图片，开始批量处理...")

        for idx, img_file in enumerate(img_list, 1):
            img_path = str(img_file.absolute())
            img_name = img_file.name
            print(f"[{idx}/{total}] 处理: {img_name}")
            try:
                img_vec = self.img_to_vec(img_path)
                data.append({"my_id": idx,
                             "name": img_name,
                             "img_vector": img_vec}
                            )
            except Exception as e:
                self.batch_failed_count += 1
                self.batch_failed_files.append((img_name, str(e)))
                print(f"❌ 处理失败: {img_name}, error={e}")

        print(f"\n📊 处理完成：成功 {self.batch_success_count} 张，失败 {self.batch_failed_count} 张")
        if self.batch_failed_files:
            print(f"❌ 失败文件列表：{[name for name, _ in self.batch_failed_files]}")
        img_embedding = data
        return img_embedding