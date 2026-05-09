"""
Freebase MID-to-readable-name resolver for semantic path scoring.

WebQSP graphs contain a mix of readable entity names ("Jamaica") and
Freebase MIDs ("m.0k8nh0b"). MiniLM cannot score paths with raw MIDs.

This module:
  1. Extracts MID→name pairs from the dataset's own graph triples
     (where a readable name co-occurs with a MID in the same triple)
  2. Provides a cacheable mapping file
  3. Falls back to the raw MID when unresolved
"""

import json
import os
import re
from typing import Dict, Optional


MID_PATTERN = re.compile(r"^m\.\d")


def is_mid(name: str) -> bool:
    """Check if a string is a Freebase MID (e.g. 'm.0k8nh0b' or '/m/0c6q0')."""
    return bool(MID_PATTERN.match(name)) or name.startswith("/m/")


def normalize_mid(name: str) -> str:
    """Normalize MID to dot format: '/m/0c6q0' → 'm.0c6q0'."""
    return name.replace("/m/", "m.")


class MidResolver:
    """
    Resolves Freebase MIDs to readable entity names.

    Usage:
        resolver = MidResolver()
        resolver.build_from_dataset(dataset)
        print(resolver.resolve("m.0k8nh0b"))   # → name or "m.0k8nh0b"

    For a complete mapping, build from a Freebase entity names file:
        resolver.build_from_fb_names_file("fb_entity_names.txt")
    """

    def __init__(self, cache_path: str = "data/mid_to_name.json"):
        self.cache_path = cache_path
        self.mid_to_name: Dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
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

    def build_from_dataset(self, dataset, id_pattern=None):
        """
        Extract MID→name pairs from dataset graph triples using heuristics:
          - If both subject and object are readable (not MIDs), they're entity-name pairs
          - Skip relation names (contain dots like 'location.country.president')
          - Use the readable name from question/answer fields when available
        """
        id_re = re.compile(r"^m\.\d")
        relation_re = re.compile(r"\w+\.\w+\.\w+")

        # Pass 1: extract from graph triples where a MID co-occurs with a readable name
        for sample in dataset:
            graph = sample.get("graph", [])
            for triple in graph:
                h, r, obj = triple

                # Skip relations
                if not relation_re.match(r):
                    continue

                # If head is MID and object is readable
                if (
                    isinstance(h, str)
                    and id_re.match(h)
                    and isinstance(obj, str)
                    and not id_re.match(obj)
                ):
                    self.mid_to_name[normalize_mid(h)] = obj

                # If object is MID and head is readable
                if (
                    isinstance(obj, str)
                    and id_re.match(obj)
                    and isinstance(h, str)
                    and not id_re.match(h)
                ):
                    self.mid_to_name[normalize_mid(obj)] = h

        # Pass 2: use answer field (readable) paired with a_entity (MID when available)
        for sample in dataset:
            answers = sample.get("answer", [])
            a_entities = sample.get("a_entity", [])
            if answers and a_entities and len(answers) == len(a_entities):
                for ans_mid, ans_text in zip(a_entities, answers):
                    if id_re.match(str(ans_mid)) and ans_text:
                        self.mid_to_name[normalize_mid(str(ans_mid))] = ans_text

        # Pass 3: use q_entity names from the question text where possible
        # (the question usually contains the readable name for q_entity MIDs)
        for sample in dataset:
            question = sample.get("question", "")
            q_entities = sample.get("q_entity", [])
            for qe in q_entities:
                if id_re.match(str(qe)) and qe not in self.mid_to_name:
                    # Try to extract from question text
                    # This is a heuristic — the question may contain the name
                    pass  # Keep the MID as fallback

        self._save()

    def download_names_file(self, target_path: str = "data/fb_entity_names.txt.gz"):
        """Download Freebase entity names from Google Cloud Storage."""
        import urllib.request

        url = "https://storage.googleapis.com/freebase-entity-names/fb_entity_names.txt.gz"
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        print(f"Downloading Freebase entity names (248 MB compressed)...")
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded to {target_path}")
        return target_path

    def build_from_fb_names_file(self, path: str, limit: Optional[int] = None):
        """
        Build mapping from a Freebase entity names file.

        Expected format (tab-separated):
            /m/0c6q0\tWarsaw

        Download:
            python -c "from dca_trie.mid_resolver import MidResolver; MidResolver().download_names_file()"
        """
        import gzip

        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mid = normalize_mid(parts[0])
                    name = parts[1]
                    self.mid_to_name[mid] = name
        self._save()

    def resolve_path_list(self, path_strs):
        """Batch resolve a list of path strings."""
        return [self.resolve_path(p) for p in path_strs]

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.mid_to_name, f, indent=1)

    def coverage(self, dataset) -> Dict:
        """Report how many MIDs in the dataset can be resolved."""
        id_re = re.compile(r"^m\.\d")
        total_mids = set()
        for sample in dataset:
            for triple in sample.get("graph", []):
                for x in [triple[0], triple[2]]:
                    if isinstance(x, str) and id_re.match(x):
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
