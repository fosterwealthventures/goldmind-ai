import os
goldmind_key = os.getenv('GOLDMIND_API_KEY')
print(f"Retrieved GOLDMIND_API_KEY: {goldmind_key}")
if goldmind_key:
    print(f"Type: {type(goldmind_key)}")
    print(f"Repr: {repr(goldmind_key)}")
    print(f"Length: {len(goldmind_key)}")
    print(f"Is ASCII: {goldmind_key.isascii()}")