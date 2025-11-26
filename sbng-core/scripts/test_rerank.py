import requests
import json
import time

def test_rerank():
    url = "http://localhost:3003/query"
    query = "electric car battery"
    
    # 1. Query WITHOUT re-ranking
    print(f"Querying: '{query}' (No Rerank)")
    start = time.time()
    res_no_rerank = requests.post(url, json={"q": query, "k": 5, "rerank": False}).json()
    dur_no_rerank = (time.time() - start) * 1000
    print(f"Time: {dur_no_rerank:.2f}ms")
    print(json.dumps(res_no_rerank, indent=2))

    # 2. Query WITH re-ranking
    print(f"\nQuerying: '{query}' (WITH Rerank)")
    start = time.time()
    res_rerank = requests.post(url, json={"q": query, "k": 5, "rerank": True}).json()
    dur_rerank = (time.time() - start) * 1000
    print(f"Time: {dur_rerank:.2f}ms")
    print(json.dumps(res_rerank, indent=2))

    # Compare top 1
    top1_no = res_no_rerank['results'][0]['doc_id'] if res_no_rerank['results'] else None
    top1_yes = res_rerank['results'][0]['doc_id'] if res_rerank['results'] else None
    
    if top1_no != top1_yes:
        print(f"\nSUCCESS: Re-ranking changed top result! ({top1_no} -> {top1_yes})")
    else:
        print(f"\nNOTE: Top result unchanged ({top1_no}). Check scores.")

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(2)
    try:
        test_rerank()
    except Exception as e:
        print(f"Error: {e}")
