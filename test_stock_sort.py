import urllib.request, urllib.parse, json

params = urllib.parse.urlencode({
    'q': '', 'category': 'CPU', 'limit': 20, 'offset': 0,
    'sort_by': 'default', 'in_stock_first': 'true'
})
url = f'http://127.0.0.1:8001/search?{params}'
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        print(f'Status: {resp.status}')
        print(f'Count: {data["count"]}')
        print('\nFirst 5:')
        for p in data['results'][:5]:
            print(f'  stock={p["stock"]}, name={p["name"][:50]}')
        print('\nLast 3:')
        for p in data['results'][-3:]:
            print(f'  stock={p["stock"]}, name={p["name"][:50]}')
        # Check if sorted correctly
        stocks = [p.get('stock') for p in data['results']]
        print(f'\nStock values in order: {stocks[:10]}...{stocks[-3:]}')
except urllib.error.HTTPError as e:
    print(f'HTTP {e.code}: {e.read().decode()}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
