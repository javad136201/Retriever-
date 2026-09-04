import requests

urls = [
    "https://api.ipify.org?format=json",
    "https://www.tsmc.com/",
]

for url in urls:
    try:
        r = requests.get(url, timeout=15)
        print("URL:", url)
        print("STATUS:", r.status_code)
        print("RESPONSE:", r.text[:500])
        print("-" * 50)
    except Exception as e:
        print("ERROR:", url, repr(e))
