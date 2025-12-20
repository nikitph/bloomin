from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def run_demo_k8s():
    # Weak Dev Manifest
    k8s_dev = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: app
        image: my-app:dev
"""
    
    # Robust Prod Manifest
    k8s_prod = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
  template:
    spec:
      containers:
      - name: app
        image: my-app:v1
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""

    print("--- Comparing Kubernetes Manifests ---")
    print("Map A: Basic Dev Deployment")
    print("Map B: Production Hardened Deployment")
    
    result = semantic_diff(k8s_dev, k8s_prod, domain="k8s")
    
    print("\nResult:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_demo_k8s()
