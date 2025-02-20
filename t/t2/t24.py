import sys
import os

# os.system("conda env list")
print("虚拟环境:", sys.executable)
for a_path in sys.path:
    print(a_path)
