import argparse
import json
import time
import requests
import os


def watch_json_files(directory: str):
    while True:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".json"):
                    print("Кошелек найден:", entry.name)
                    return True
            print("result not found...")
        time.sleep(0.1)


parser = argparse.ArgumentParser(description="Enqueue vanity search task")
parser.add_argument('--prefix', type=str, required=False, default="SoL", help='Префикс (можно несколько символов)')
parser.add_argument('--suffix', type=str, required=False, default="", help='Суффикс')
parser.add_argument('--case-sensitive', action='store_true', help='Учитывать регистр')

args = parser.parse_args()

url = "http://localhost:5000/enqueue"

headers = {
    'Content-Type': 'application/json'
}

payload = json.dumps({
    "prefix": list(args.prefix),
    "suffix": args.suffix,
    "case_sensitive": args.case_sensitive
})

response = requests.post(
    url=url,
    headers=headers,
    data=payload
)
response.raise_for_status()

print(response.text)
print("Запускаем ожидание кошелька...")

watch_json_files(os.path.dirname(os.path.abspath(__file__)))




