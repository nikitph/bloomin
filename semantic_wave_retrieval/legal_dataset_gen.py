import wikipediaapi
import random
import json

DOCTRINES = [
    "At-will employment",
    "Independent contractor",
    "Consideration", 
    "Force majeure",
    "Negligence",
    "Due process"
]

# Hardcoded queries for consistency in this script, or we can use LLM to generate more.
# For MVP, let's define 3 distinct queries per doctrine => 18 base queries.
# Then expand to 4 variants each => 72 queries.
QUERIES_BASE = {
    "At-will employment": [
        "Can an employer fire an employee without reason?",
        "What are exceptions to at-will employment?",
        "Is notice required for termination in at-will states?"
    ],
    "Independent contractor": [
        "What is the difference between an employee and a contractor?",
        "Does a contractor get benefits like health insurance?",
        "What is the ABC test for independent contractors?"
    ],
    "Consideration": [
        "Is a contract valid without consideration?",
        "What constitutes valid consideration in a contract?",
        "Can a gift be enforced as a contract?"
    ],
    "Force majeure": [
        "What events trigger a force majeure clause?",
        "Can a pandemic excuse contract performance?",
        "Is force majeure implied in all contracts?"
    ],
    "Negligence": [
        "What are the four elements of negligence?",
        "How is duty of care determined?",
        "What is comparative negligence?"
    ],
    "Due process": [
        "What does procedural due process guarantee?",
        "Is notice and a hearing required for deprivation of property?",
        "Does due process apply to private companies?"
    ]
}

# New Complex Queries for DAS Metric
# Format: Query -> {Required, Allowed, Forbidden}
QUERIES_COMPLEX = [
    {
        "query": "Can consideration be waived in force majeure events?",
        "type": "Cross-Doctrine",
        "required": ["Consideration", "Force majeure"],
        "allowed": [],
        "forbidden": ["Negligence", "Due process"]
    },
    {
        "query": "Is an independent contractor entitled to due process in termination?",
        "type": "Cross-Doctrine",
        "required": ["Independent contractor", "Due process"],
        "allowed": ["At-will employment"],
        "forbidden": ["Force majeure"]
    },
    {
        "query": "Can an at-will employee be fired for whistleblowing?",
        "type": "Exception",
        "required": ["At-will employment"],
        "allowed": ["Negligence"], # Whistleblowing often relates to reporting negligence/safety
        "forbidden": ["Force majeure", "Consideration"]
    },
    {
        "query": "Does negligence theory apply to breach of contract?",
        "type": "Cross-Doctrine",
        "required": ["Negligence", "Consideration"], # Contract = Consideration context
        "allowed": [],
        "forbidden": ["Due process"]
    }
]

def generate_variants(query):
    # Mocking variants generation for MVP reproducibility without valid LLM key for generation
    # In real usage, use LLM to paraphrase/adversarial.
    # Here, heuristic rules.
    
    clean = query
    
    # Paraphrase (Mock)
    words = query.split()
    if len(words) > 3:
        para = f"Regarding {words[-1]}, {query.lower()}?" 
    else:
        para = query + " paraphrased"
    
    # OCR (Leet speak / Typos)
    ocr = query.replace('o', '0').replace('e', '3').replace('i', '1').replace('a', '4')
    
    # Adversarial (Append distractor)
    adv = query + " in context of quantum mechanics and gardening"
    
    return {
        "Clean": clean,
        "Paraphrased": para,
        "OCR": ocr,
        "Adversarial": adv
    }

class LegalDatasetGenerator:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('BloominLegalBot/1.0', 'en')
        
    def generate(self):
        corpus = []
        labels = []
        
        print("Fetching doctrines...")
        for doctrine in DOCTRINES:
            page = self.wiki.page(doctrine)
            if not page.exists():
                print(f"Skipping {doctrine}")
                continue
                
            text = page.text
            # Split into chunks
            chunks = [p for p in text.split('\n') if len(p) > 100]
            # Take top 50
            chunks = chunks[:50]
            
            for c in chunks:
                corpus.append(c)
                labels.append(doctrine)
                
            print(f"Loaded {len(chunks)} chunks for {doctrine}")
            
        print(f"Total Corpus Size: {len(corpus)}")
        
        # Generate Queries
        benchmark_set = []
        
        # 1. Base Queries
        for doctrine, qs in QUERIES_BASE.items():
            for q in qs:
                variants = generate_variants(q)
                entry = {
                    "type": "Base",
                    "ground_truth_basin": doctrine,
                    "queries": variants,
                    "das_criteria": {
                        "required": [doctrine],
                        "allowed": DOCTRINES, # Lax for base
                        "forbidden": []
                    }
                }
                benchmark_set.append(entry)
        
        # 2. Complex Queries (DAS Optimized)
        for item in QUERIES_COMPLEX:
            q = item['query']
            variants = generate_variants(q)
            entry = {
                "type": item['type'],
                "queries": variants,
                "das_criteria": {
                    "required": item['required'],
                    "allowed": item['allowed'] + item['required'],
                    "forbidden": item['forbidden']
                }
            }
            benchmark_set.append(entry)
                
        return corpus, labels, benchmark_set

if __name__ == "__main__":
    gen = LegalDatasetGenerator()
    corpus, labels, bench_set = gen.generate()
    # Save to file? Or just print for now.
    import pickle
    with open("legal_benchmark_data.pkl", "wb") as f:
        pickle.dump((corpus, labels, bench_set), f)
    print("Saved legal_benchmark_data.pkl")
