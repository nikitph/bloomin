# Quick Start Guide

## Installation (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run the System (1 minute)

```bash
# Process sample logs
python main.py
```

That's it! The system will:
1. Load 6 CVE signatures (Log4Shell, EternalBlue, BlueKeep, Zerologon, ProxyLogon, FortiOS)
2. Process 25 sample logs
3. Detect multi-stage APT campaigns
4. Generate tiered alerts
5. Export results to `data/results/`

## View Results

Check the output files:

```bash
# View alerts
cat data/results/alerts.json | python -m json.tool | less

# View metrics
cat data/results/metrics.json | python -m json.tool

# View campaigns
cat data/results/campaigns.json | python -m json.tool
```

## Example Output

```
✓ Processed 25 logs
✓ Generated 24 alerts
✓ Detected 39 multi-stage attacks
✓ Identified 14 APT campaigns
✓ Memory usage: 31.17 MB (under 720 MB budget)
```

## What Just Happened?

The system detected several APT campaigns in the sample data:

1. **ProxyLogon Attack** (CVE-2021-26855):
   - 5-stage attack over 20.4 days
   - TIER 3 CRITICAL alert

2. **EternalBlue Campaign** (CVE-2017-0144):
   - 4-stage attack over 20.5 days
   - TIER 3 CRITICAL alert

3. **Zerologon Exploitation** (CVE-2020-1472):
   - 3-stage attack over 3.3 days
   - TIER 3 CRITICAL alert

## Next Steps

### 1. Process Your Own Logs

Create a JSON file with your logs:

```json
[
  {
    "timestamp": "2024-11-04T10:00:00Z",
    "source_ip": "10.0.1.5",
    "target": "server-01",
    "log": "Your log message here"
  }
]
```

Run with your logs:

```bash
python main.py --logs /path/to/your/logs.json
```

### 2. Add Custom CVE Signatures

Edit `data/signatures/cve_signatures.json` to add new CVE patterns.

### 3. Tune Configuration

Edit `config.yaml` to adjust:
- Bloom filter sizes
- Detection thresholds
- Memory limits
- Performance settings

### 4. Run Tests

```bash
# Run correlation tests
pytest tests/test_correlation.py -v

# Run performance benchmarks
pytest tests/test_performance.py -v

# Run all tests
pytest tests/ -v
```

## Understanding the Output

### Alert Tiers

- **Tier 1 (LOW)**: Single stage detected - possible reconnaissance
- **Tier 2 (HIGH)**: 2-stage attack detected - active threat
- **Tier 3 (CRITICAL)**: 3+ stages detected - confirmed APT campaign

### Key Metrics

- **Throughput**: Logs processed per second
- **Latency**: Processing time per log (μs)
- **Memory**: Current memory usage
- **Correlations**: Multi-stage attacks detected

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | >200K logs/sec | Single-threaded |
| Latency | <5 μs per log | Pattern matching |
| Memory | <720 MB | Constant, not growing |

Note: Current implementation processes ~300 logs/sec with ~3ms per log. This is because we're using Python with full safety checks. Production C++ implementation would hit 200K+ logs/sec targets.

## Common Use Cases

### Scenario 1: Real-time Log Monitoring

Stream logs into the system in real-time:

```python
from src.processors.log_processor import LogProcessor
from src.signatures.cve_loader import CVELoader
from src.bloom.bloom_config import BloomConfig

# Initialize
cve_loader = CVELoader("data/signatures/cve_signatures.json")
processor = LogProcessor(cve_loader)

# Process streaming logs
for log in log_stream:
    alert = processor.process_log(log)
    if alert and alert.tier >= 3:
        send_critical_alert(alert)
```

### Scenario 2: Historical Log Analysis

Process large historical log files:

```bash
python main.py --logs historical_logs.json --output results/historical
```

### Scenario 3: Custom CVE Detection

Add your organization's custom threat signatures:

1. Edit `data/signatures/cve_signatures.json`
2. Add new CVE with stages and patterns
3. Map to MITRE ATT&CK techniques
4. Run: `python main.py`

## Troubleshooting

### Issue: "Module not found"

```bash
pip install -r requirements.txt
```

### Issue: Low throughput

- Current Python implementation is ~300 logs/sec
- For production: implement core in C/C++
- Enable parallel processing in config.yaml

### Issue: Memory growing

- Check Bloom filter configuration
- Verify error rates are reasonable
- Monitor memory with: `python main.py --verbose`

## Support

- Read the full [README.md](README.md)
- Check [tests/](tests/) for examples
- Review CVE signatures in [data/signatures/](data/signatures/)

---

**Ready to detect APT campaigns!** 🚀
