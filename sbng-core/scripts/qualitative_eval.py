import requests
import json

SBNG_URL = "http://localhost:3001/query"

queries = [
    "electric car battery",
    "apple fruit",
    "python programming language",
    "bank river water",
    "jaguar animal speed"
]

def run_eval():
    print(f"Qualitative Evaluation on {len(queries)} queries...\n")
    
    for q in queries:
        print(f"Query: '{q}'")
        
        # 1. SBNG (No Rerank)
        try:
            res_no = requests.post(SBNG_URL, json={"q": q, "k": 3, "rerank": False}).json()
            top_no = [(r['doc_id'], r['score']) for r in res_no.get('results', [])]
        except:
            top_no = []
            
        # 2. SBNG + Rerank
        try:
            res = requests.post(SBNG_URL, json={"q": q, "k": 3, "rerank": True})
            if res.status_code != 200:
                print(f"  Error: {res.status_code} {res.text}")
                top_yes = []
            else:
                res_yes = res.json()
                top_yes = [(r['doc_id'], r['score']) for r in res_yes.get('results', [])]
        except Exception as e:
            print(f"  Exception: {e}")
            top_yes = []
            
        print("  SBNG Top-3:   ", top_no)
        print("  Rerank Top-3: ", top_yes)
        
        # Check if order changed
        ids_no = [x[0] for x in top_no]
        ids_yes = [x[0] for x in top_yes]
        
        if ids_no != ids_yes:
            print("  -> ORDER CHANGED: Yes")
            if ids_no and ids_yes and ids_no[0] != ids_yes[0]:
                 print(f"  -> TOP RESULT CHANGED: {ids_no[0]} -> {ids_yes[0]}")
        else:
            print("  -> ORDER CHANGED: No")
        print("-" * 40)

if __name__ == "__main__":
    run_eval()
