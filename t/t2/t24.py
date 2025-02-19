import sys
print("虚拟环境", sys.executable)
for a_path in sys.path:
    print(a_path)
