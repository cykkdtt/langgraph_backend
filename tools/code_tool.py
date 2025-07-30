from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import json
import matplotlib
matplotlib.use('Agg')
import os
import glob

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        
        # 检查是否生成了图片文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif']:
            image_files.extend(glob.glob(ext))
        
        # 将图片移动到前端可访问的目录
        if image_files:
            import shutil
            for img_file in image_files:
                if os.path.exists(img_file):
                    # 确保目标目录存在
                    target_dir = "agent-chat-ui/public/images"
                    os.makedirs(target_dir, exist_ok=True)
                    # 移动文件
                    target_path = os.path.join(target_dir, img_file)
                    shutil.move(img_file, target_path)
                    # 更新图片文件列表为移动后的路径
                    image_files = [os.path.join("images", img_file) for img_file in image_files]
        
        if image_files:
            # 将图片转换为 base64 格式
            import base64
            for img_file in image_files:
                # 获取移动后的完整路径
                full_path = os.path.join("agent-chat-ui/public", img_file)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            img_data = f.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            # 根据文件扩展名确定 MIME 类型
                            filename = os.path.basename(img_file)
                            mime_type = 'image/png' if filename.endswith('.png') else \
                                       'image/jpeg' if filename.endswith(('.jpg', '.jpeg')) else \
                                       'image/gif' if filename.endswith('.gif') else 'image/png'
                            
                            # 创建图片数据块
                            image_block = {
                                "type": "image",
                                "source_type": "base64", 
                                "mime_type": mime_type,
                                "data": img_base64,
                                "metadata": {"name": filename}
                            }
                            
                            # 将图片数据添加到结果中
                            result += f"\n\n📊 生成的图片: {filename}"
                            result += f"\n[IMAGE_DATA:{json.dumps(image_block)}]"
                    except Exception as e:
                        result += f"\n\n⚠️ 处理图片 {filename} 时出错: {str(e)}"
            
            result += "\n\n图片已生成并包含在响应中。"
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n\\`\\`\\`python\n{code}\n\\`\\`\\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )