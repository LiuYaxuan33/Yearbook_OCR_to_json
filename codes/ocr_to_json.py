# 此脚本用于将图片信息直接转为json格式

from paddleocr import PaddleOCR
import requests
from openai import OpenAI
import json
import base64
import re
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
# 初始化 OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    det_model_dir='en_PP-OCRv3_det',
    rec_model_dir='en_PP-OCRv3_rec',
    cls_model_dir='ch_ppocr_mobile_v2.0_cls'
)

#这部分不要改
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"
#这部分的路径改成实际路径
IMAGE_FOLDER = ''
OUTPUT_FILE = 'output.json'

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ocr_with_paddle(image_path):
    result = ocr.ocr(image_path, cls=True)
    return "\n".join([word_info[1][0] for line in result for word_info in line])

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
Read all the text in the image, preserving original line breaks.
"""},
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

    all_students = []

    with open(output_file.replace('.json', '.txt'), "w", encoding="utf-8") as txt_file:
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"正在处理 ({idx}/{len(image_files)}): {os.path.basename(image_path)}")
                
                # OCR处理
                #ocr_text = ocr_with_paddle(image_path)
                ocr_text = ocr_with_qwen(image_path)
                #print(ocr_text)
                # 文本润色
                rewritten_text = deepseek_rewrite(ocr_text)
                #print("z")
                #print(rewritten_text)
                # 格式转换
                student_json = deepseek_to_json(rewritten_text)
                if student_json:
                    all_students.extend(student_json)

                # 写入文本结果
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")

    # 写入JSON结果
    if all_students:
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(all_students, json_file, ensure_ascii=False, indent=2)
        print(f"\nJSON结果已保存至：{os.path.abspath(output_file)}")

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
Perform text refinement with these priorities:
1. Fundamental Corrections:
   - Capitalize proper nouns (e.g.: john doe → John Doe, ARNOLD → Arnold)
   - Fix letter confusions (e.g.: cl → d, rn → m, 1 → l， 0 → o)
   - Correct obvious spelling errors (e.g.: hel1o → hello)
   - Complete missing initials of common surnames (e.g.: pple → Apple, Arry → Barry)

2. Paragraph Optimization:
   - Merge erroneous line breaks (e.g.: "Please submit\nthe report" → "Please submit the report")
   - Preserve intentional paragraph breaks (e.g.: section headings)
   - Reorganize paragraphs based on semantic coherence

3. Output Requirements:
   - Maintain original information integrity
   - Return only the polished text
   - Use standard English punctuation

Example:
[Raw OCR]
Project proqress Report:
As of this week, Team A has completed
80% of the tasl. The main chal1enge
is data co1lection delays. Team B
progress is s1ower than expected.

[Refined]
Project Progress Report:
As of this week, Team A has completed 80% of the task. 
The main challenge is data collection delays. 
Team B progress is slower than expected.

Provide only the pure text version of the polished text, do not add extra words of markdown notation.

Delete anything that look like pagination notations.
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

def deepseek_to_json(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": """
请严格按以下规则解析学生信息为JSON格式：
1. 字段顺序必须为：name, gender, hometown, nickname, major, clubs, comment
2. 姓名格式规则：
   - 检查姓氏，如果和常见姓氏相比缺失首字母（如Barry -> Arry）则补全
   - 每个单词的首字母大写，其余字母小写
   - 姓氏在前，名在后，二者以逗号隔开（即常见的排版方式）
3. 性别推断基于英文名常见性别（Male/Female）
4. 籍贯识别：
   - 匹配已知国家/城市名格式（如New York, London）
5. 昵称识别：
   - 常在双引号中
   - 保留原有昵称
   - 将所有昵称（如有）以列表形式储存
6. 专业识别：
   - 保留原有专业名称或缩写（如H. Ec., Vet., C. E., Agr.等）
7. 社团信息：
   - 不成句的单词/词组
   - 每行一个条目
   - 没有则留空列表
   - 社团与社团之间常用连字符'-'隔开
   - 包含可能的college或university信息
8. 评语：
   - 完整的句子
   - 去除原始换行符
   - 保留原有标点符号，如有需要，使用转义符
9. 保留原始信息，除非明显错误

示例输入：
LBECHT, L.R. C.E.
Tama, Iowa  
“dad”
Pi Beta Phi-Philomathean

When joy and duty clash, let duty go to smash. Ames found a loyal convert in Louise after sojourns at Leander Clark and Cornell. A great student at times. Prefers Domestic Science because of its excellent home training. A girl with whom to have the best kind of a time.  

示例输出：
{
  "name": "Albrecht, L. R.",
  "gender": "Male",
  "major": "C.E.",
  "nicknames": ["dad"],
  "hometown": "Tama, Iowa",
  "clubs": ["Pi Beta Phi","Philomathean"],
  "comment": "When joy and duty clash, let duty go to smash. Ames found a loyal convert in Louise after sojourns at Leander Clark and Cornell. A great student at times. Prefers Domestic Science because of its excellent home training. A girl with whom to have the best kind of a time."
}

只需返回JSON，不要添加任何代码块标记（如markdown格式的```json），只返回纯文本，也不要额外解释。把所有学生信息放在同一个列表中。
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
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        json_str = response.json()['choices'][0]['message']['content'].strip()
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        # print(json_str)
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON转换失败: {str(e)}")
        print(json_str)
        return None

# 使用示例
if __name__ == "__main__":
    image_folder = IMAGE_FOLDER
    output_file = OUTPUT_FILE
    
    process_image_folder(image_folder, output_file)