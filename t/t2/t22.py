from dotenv import load_dotenv
import os
load_dotenv()
print(36)
# 获取 PYTHONPATH 环境变量的值
pythonpath_value = os.getenv('PYTHONPATH')
pythonpath_value2 = os.getenv('PYTHONPATH2')
pythonpath_value3 = os.getenv('PYTHONPATH3')

# 打印值
print(f"PYTHONPATH: {pythonpath_value}")
print(f"PYTHONPATH2: {pythonpath_value3}")
print(f"PYTHONPATH3: {pythonpath_value}")
