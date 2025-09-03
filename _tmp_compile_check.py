import py_compile
import sys

files = ["src/unified_chat/chat_service.py", "src/api/chat_endpoints.py"]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print("OK", f)
    except Exception as e:
        print("ERR", f, e)
        sys.exit(1)
print("syntax ok")
