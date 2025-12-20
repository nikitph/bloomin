import yaml

def extract_witnesses_k8s(manifest_yaml: str) -> set[str]:
    """
    Extracts semantic witnesses from Kubernetes YAML manifests.
    """
    witnesses = set()
    try:
        # Load all documents in case of multi-doc YAML
        docs = yaml.safe_load_all(manifest_yaml)
    except yaml.YAMLError:
        print("Warning: Invalid YAML for K8s manifest")
        return witnesses

    for doc in docs:
        if not doc: continue
        
        kind = doc.get("kind", "")
        if kind:
            witnesses.add(kind)
            
        spec = doc.get("spec", {})
        
        # Check replicas for HA
        if kind == "Deployment" or kind == "StatefulSet":
            replicas = spec.get("replicas", 1)
            if replicas > 1:
                witnesses.add("replicas_ha")
            
            # Strategy
            strategy = spec.get("strategy", {})
            if strategy.get("type") == "RollingUpdate":
                witnesses.add("strategy_rolling_update")
                
        # Pod Spec (embedded in Deployment or Pod)
        template_spec = spec
        if kind == "Deployment":
             template_spec = spec.get("template", {}).get("spec", {})
             
        containers = template_spec.get("containers", [])
        for c in containers:
            # Probes
            if "livenessProbe" in c: witnesses.add("liveness_probe")
            if "readinessProbe" in c: witnesses.add("readiness_probe")
            
            # Resources
            resources = c.get("resources", {})
            limits = resources.get("limits", {})
            requests = resources.get("requests", {})
            
            if "cpu" in limits: witnesses.add("cpu_limit")
            if "memory" in limits: witnesses.add("memory_limit")
            if "cpu" in requests: witnesses.add("cpu_request")
            if "memory" in requests: witnesses.add("memory_request")
            
            # Security Context
            security = c.get("securityContext", {})
            if security.get("privileged"): witnesses.add("privileged")
            if security.get("runAsUser") == 0: witnesses.add("run_as_root")
            if security.get("readOnlyRootFilesystem"): witnesses.add("read_only_root_filesystem")
            
        # Strict Recursive Check for Encryption
        # Avoids matching labels/annotations or comments
        if find_encryption_recursive(spec):
             witnesses.add("encryption_at_rest")
            
    return witnesses

def find_encryption_recursive(node):
    if isinstance(node, dict):
        for k, v in node.items():
            # Check for key "encryption" with value True (or loop deeper)
            if k.lower() == "encryption":
                if v is True or (isinstance(v, str) and v.lower() == "true"):
                    return True
            if find_encryption_recursive(v):
                return True
    elif isinstance(node, list):
        for item in node:
            if find_encryption_recursive(item):
                return True
    return False
