import argparse
import json
import requests

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

response = requests.post(url, headers=headers, data=payload)
response.raise_for_status()

print(response.status_code)
print(response.text)
