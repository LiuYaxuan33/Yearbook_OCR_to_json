
import requests
from openai import OpenAI
import json
import base64
from dotenv import load_dotenv
import os

load_dotenv()  # 加载.env文件中的环境变量

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
MODEL_NAME = "deepseek-chat"
#这部分的路径改成实际路径
IMAGE_FOLDER = 'output_images'
OUTPUT_FILE = 'output/output.json'

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def ocr_with_qwen(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    img_format = image_path.split('.')[-1].lower()
    mime_type = f"image/{img_format}" if img_format in ['png', 'jpeg', 'jpg', 'webp'] else "image/jpeg"

    completion = client.chat.completions.create(
        model="qwen-vl-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": 
"""
请严格按照以下要求执行OCR识别：
1. 全面扫描：
   - 按从左到右、从上到下的顺序逐行扫描
   - 确保每行都完整识别从最左端到最右端的内容
   - 特别注意行末可能被截断的单词
   - 特别注意靠右对齐的零散内容如“Ames, Ia.”或“C.E.”
   - 自上而下逐行识别，在同一行的内容（不管中间有多大空白）全部识别完成后再识别下一行。

2. 布局保留：
   - 严格保持原始换行符（包括空行）
   - 行内连续文本不要擅自添加换行
   - 保留段落之间的自然空行

3. 特殊处理：
   - 连字符结尾的行（如"electro-")要与下一行衔接
   - 识别表格/分栏内容时，按视觉顺序而非逻辑顺序
   - 数字0和字母O要结合上下文区分

4. 完整性检查：
   - 如果行末字符靠近图片边缘，需二次确认
   - 对模糊区域采用概率采样而非直接跳过
"""
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content.strip()

def process_image_folder(folder_path, output_file):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        print("未找到支持的图片文件")
        return

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 新增合并文件路径
    merged_ocr_path = os.path.join(output_dir, "merged_ocr_results.txt")
    merged_rewritten_path = os.path.join(output_dir, "merged_rewritten_results.txt")
    

    with open(merged_ocr_path, "w", encoding="utf-8") as ocr_merged_file, \
         open(merged_rewritten_path, "w", encoding="utf-8") as rewritten_merged_file:

        for idx, image_path in enumerate(image_files, 1):
            try:
                image_name = os.path.basename(image_path)
                print(f"正在处理 ({idx}/{len(image_files)}): {image_name}")

                # OCR处理
                ocr_text = ocr_with_qwen(image_path)
                
                # 写入合并OCR文件
                ocr_merged_file.write(ocr_text + "\n\n")

                # 文本润色
                rewritten_text = deepseek_rewrite(ocr_text)
                
                # 写入合并润色文件
                rewritten_merged_file.write(rewritten_text)
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")


        
def deepseek_rewrite(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                 "content": 
"""
Perform OCR fusion and text refinement with these priorities:

Chinese Character Recognition:
Keep the original Chinese characters and do not delete them
Comprehensive Corrections:
Resolve character confusions
Complete partial words
Fix line-break errors while preserving valid paragraph breaks
Insert paragraph spacing between distinct information blocks (e.g., store profiles)
Output Requirements:
Return only the enhanced text
Preserve all informational elements
Never add explanatory comments
Provide only the pure polished text without any markdown format (‘’‘, *, etc.)
If the input begins with a store name, leave a line empty at the beginning.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API请求失败，返回原始文本。错误信息: {str(e)}")
        return text


# 使用示例
if __name__ == "__main__":
    image_folder = IMAGE_FOLDER
    output_file = OUTPUT_FILE
    process_image_folder(image_folder, output_file)