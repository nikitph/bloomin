from witness_arithmetic_mvp.api_extractor import extract_witnesses_api
from witness_arithmetic_mvp.encoder import encode
from witness_arithmetic_mvp.arithmetic import load_vocabulary, decode

api_v1 = """
{
  "openapi": "3.0.0",
  "paths": {
    "/users": {
        "get": { "summary": "List users" }
    }
  }
}
"""

api_v2 = """
{
  "openapi": "3.0.0",
  "security": [{"ApiKeyAuth": []}],
  "paths": {
    "/users": {
        "post": {
            "summary": "Create user",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "email": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
  }
}
"""

w1 = extract_witnesses_api(api_v1)
w2 = extract_witnesses_api(api_v2)

print(f"W1: {w1}")
print(f"W2: {w2}")

b1 = encode(w1)
b2 = encode(w2)

vocab = load_vocabulary("api")
print(f"Vocab size: {len(vocab)}")
if "pii_email" not in vocab:
    print("ERROR: pii_email NOT in vocab")
else:
    print("pii_email is in vocab")

# Check if pii_email bits are in b1
w_pii = {"pii_email"}
b_pii = encode(w_pii)

print(f"PII Bits: {b_pii.to01()}")
print(f"B1  Bits: {b1.to01()}")

overlap = b_pii & b1
print(f"Overlap:  {overlap.to01()}")

if overlap == b_pii:
    print("COLLISION: PII Email bits are fully present in B1 witnesses!")
else:
    print("No full collision.")

# Check delta
delta = b1 ^ b2
added_bits = delta & b2
decoded_added = decode(added_bits, vocab)
print(f"Decoded Added: {decoded_added}")
