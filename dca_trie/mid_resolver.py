"""
Freebase MID-to-readable-name resolver for semantic path scoring.

WebQSP graphs contain a mix of readable entity names ("Jamaica") and
Freebase MIDs ("m.0k8nh0b"). MiniLM cannot score paths with raw MIDs.

This module:
  1. Extracts MID→name pairs from the dataset's own graph triples
     (where a readable name co-occurs with a MID in the same triple)
  2. Extracts MID→name pairs from question/answer text (Pass 3)
  3. Downloads the complete Freebase entity names file for ~99% coverage
  4. Provides a cacheable mapping file
  5. Falls back to the raw MID when unresolved
"""

import json
import os
import re
from typing import Dict, Optional, List

_MID_DOT = re.compile(r"^m\.\d")
_MID_SLASH = re.compile(r"^/m/\d")
_RELATION = re.compile(r"\w+\.\w+\.\w+")


def is_mid(name: str) -> bool:
    """Check if a string is a Freebase MID (e.g. 'm.0k8nh0b' or '/m/0c6q0')."""
    return bool(_MID_DOT.match(name)) or bool(_MID_SLASH.match(name))


def normalize_mid(name: str) -> str:
    """Normalize MID to dot format: '/m/0c6q0' → 'm.0c6q0' or 'm.0c6q0' → 'm.0c6q0'."""
    return name.replace("/m/", "m.") if name.startswith("/m/") else name


class MidResolver:
    """
    Resolves Freebase MIDs to readable entity names.

    Usage:
        resolver = MidResolver()
        resolver.build_from_dataset(dataset)
        print(resolver.resolve("m.0k8nh0b"))   # → name or "m.0k8nh0b"

    For near-complete mapping (~99%):
        resolver.download_and_build_fb_names()
    """

    def __init__(self, cache_path: str = "data/mid_to_name.json"):
        self.cache_path = os.path.abspath(cache_path) if not os.path.isabs(cache_path) else cache_path
        self.mid_to_name: Dict[str, str] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self.mid_to_name = json.load(f)

    def resolve(self, name: str) -> str:
        """If name is a MID, return readable name; otherwise return as-is."""
        if not is_mid(name):
            return name
        normalized = normalize_mid(name)
        return self.mid_to_name.get(normalized, name)

    def resolve_path(self, path_str: str) -> str:
        """Replace all MIDs in a path string with readable names."""
        parts = path_str.split(" -> ")
        resolved = [self.resolve(p) for p in parts]
        return " -> ".join(resolved)

    def resolve_path_list(self, path_strs: List[str]) -> List[str]:
        """Batch resolve a list of path strings."""
        return [self.resolve_path(p) for p in path_strs]

    def build_from_dataset(self, dataset):
        """
        Extract MID→name pairs from dataset graph triples using heuristics.

        Three passes:
          Pass 1: Graph triples where a MID co-occurs with a readable name.
          Pass 2: Answer field (readable) paired with a_entity (MID).
          Pass 3: Question text — extract entity names for q_entity MIDs.
        """
        new_entries = 0

        # Pass 1: extract from graph triples
        for sample in dataset:
            graph = sample.get("graph", [])
            for triple in graph:
                h, r, obj = triple
                if not _RELATION.match(r):
                    continue
                if isinstance(h, str) and _MID_DOT.match(h) and isinstance(obj, str) and not _MID_DOT.match(obj):
                    mid = normalize_mid(h)
                    if mid not in self.mid_to_name:
                        self.mid_to_name[mid] = obj
                        new_entries += 1
                if isinstance(obj, str) and _MID_DOT.match(obj) and isinstance(h, str) and not _MID_DOT.match(h):
                    mid = normalize_mid(obj)
                    if mid not in self.mid_to_name:
                        self.mid_to_name[mid] = h
                        new_entries += 1

        if new_entries:
            print(f"  Pass 1 (graph triples): +{new_entries} entries")

        # Pass 2: answer + a_entity pairs
        new_entries = 0
        for sample in dataset:
            answers = sample.get("answer", [])
            a_entities = sample.get("a_entity", [])
            if answers and a_entities and len(answers) == len(a_entities):
                for ans_mid, ans_text in zip(a_entities, answers):
                    if _MID_DOT.match(str(ans_mid)) and ans_text:
                        mid = normalize_mid(str(ans_mid))
                        if mid not in self.mid_to_name:
                            self.mid_to_name[mid] = str(ans_text)
                            new_entries += 1

        if new_entries:
            print(f"  Pass 2 (answer entities): +{new_entries} entries")

        # Pass 3: q_entity names from question text
        new_entries = 0
        for sample in dataset:
            question = sample.get("question", "")
            q_entities = sample.get("q_entity", [])
            for qe in q_entities:
                if _MID_DOT.match(str(qe)):
                    mid = normalize_mid(str(qe))
                    if mid not in self.mid_to_name:
                        name = _extract_entity_from_question(question, qe, dataset)
                        if name:
                            self.mid_to_name[mid] = name
                            new_entries += 1

        if new_entries:
            print(f"  Pass 3 (question text): +{new_entries} entries")

        self._save()

    def download_and_build_fb_names(
        self,
        url: Optional[str] = None,
        target_path: str = "data/fb_entity_names.txt.gz",
        limit: Optional[int] = None,
    ):
        """Download Freebase names and build mapping in one call."""
        actual_url = url or "https://storage.googleapis.com/freebase-entity-names/fb_entity_names.txt.gz"
        self.download_names_file(actual_url, target_path)
        before = len(self.mid_to_name)
        self.build_from_fb_names_file(target_path, limit=limit)
        print(f"  Loaded {len(self.mid_to_name) - before} new entries from Freebase names file")

    def download_names_file(self, url: str, target_path: str = "data/fb_entity_names.txt.gz"):
        """Download Freebase entity names file."""
        import urllib.request
        target = os.path.abspath(target_path) if not os.path.isabs(target_path) else target_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading Freebase entity names (248 MB compressed)...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, target)
        print(f"  Downloaded to {target}")
        return target

    def build_from_fb_names_file(self, path: str, limit: Optional[int] = None):
        """Build mapping from a Freebase entity names file (tab-separated: /m/xxx\tName)."""
        import gzip
        open_fn = gzip.open if path.endswith(".gz") else open
        count = 0
        with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 2 and _MID_DOT.match(normalize_mid(parts[0])):
                    mid = normalize_mid(parts[0])
                    name = parts[1].strip()
                    if name and len(name) < 200:
                        self.mid_to_name[mid] = name
                        count += 1
        print(f"  Loaded {count} MID→name entries")
        self._save()

    def _save(self):
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.mid_to_name, f, indent=1)

    def coverage(self, dataset) -> Dict:
        """Report how many MIDs in the dataset can be resolved."""
        total_mids = set()
        for sample in dataset:
            for triple in sample.get("graph", []):
                for x in [triple[0], triple[2]]:
                    if isinstance(x, str) and _MID_DOT.match(x):
                        total_mids.add(normalize_mid(x))

        resolved = sum(1 for m in total_mids if m in self.mid_to_name)
        return {
            "total_mids": len(total_mids),
            "resolved": resolved,
            "unresolved": len(total_mids) - resolved,
            "coverage_pct": round(resolved / len(total_mids) * 100, 1)
            if total_mids
            else 0,
        }


def _extract_entity_from_question(question: str, q_entity_mid: str, dataset) -> Optional[str]:
    """
    Extract readable entity name from question text for a q_entity MID.

    Strategy: the question entity is usually the first noun phrase.
    We look for the first multi-word sequence that isn't a stopword/verb.
    """
    if not question:
        return None
    stop_words = {"what", "where", "when", "who", "which", "how", "is", "are",
                  "was", "were", "did", "do", "does", "the", "a", "an", "of",
                  "in", "on", "at", "to", "for", "with", "by", "from", "and", "or"}

    words = question.strip().rstrip("?").split()

    # Skip leading stop words
    start = 0
    while start < len(words) and words[start].lower() in stop_words:
        start += 1

    if start >= len(words):
        return None

    # Take consecutive non-stop words from the start
    entity_words = []
    for i in range(start, len(words)):
        if words[i].lower() in stop_words and entity_words:
            break
        if words[i][0].isupper() or not entity_words:
            entity_words.append(words[i])
        elif entity_words:
            break

    name = " ".join(entity_words) if entity_words else None
    if name and len(name) > 3 and len(name) < 100:
        return name
    return None
