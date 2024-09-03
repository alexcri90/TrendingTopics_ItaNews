import sys
import os

print("Python executable:", sys.executable)
print("\nPython version:", sys.version)
print("\nsys.path:")
for path in sys.path:
    print(path)

print("\nCurrent working directory:", os.getcwd())
print("\nContents of current directory:", os.listdir())

try:
    import requests
    print("\nrequests module is installed and can be imported.")
except ImportError:
    print("\nFailed to import requests module.")