"""Exact-match glossary retrieval for biomedical terms."""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


class GlossaryRetriever:
    """Retrieves biomedical term definitions using exact phrase matching."""

    def __init__(self, glossary_path: str):
        """
        Initialize the glossary retriever.

        Args:
            glossary_path: Path to CSV with 'word' and 'definition' columns
        """
        self.glossary_path = Path(glossary_path)
        self.glossary: Dict[str, str] = {}
        self.term_patterns: List[Tuple[str, re.Pattern]] = []
        self._load_glossary()

    def _normalize_term(self, term: str) -> str:
        """Normalize a term for matching: lowercase and clean whitespace."""
        return " ".join(term.lower().split())

    def _load_glossary(self):
        """Load and deduplicate glossary, keeping shortest definitions."""
        df = pd.read_csv(self.glossary_path)
        
        if 'word' not in df.columns or 'definition' not in df.columns:
            raise ValueError(
                f"Glossary CSV must have 'word' and 'definition' columns. "
                f"Found: {df.columns.tolist()}"
            )
        
        # Deduplicate: keep shortest definition per normalized term
        term_defs: Dict[str, List[str]] = {}
        
        for _, row in df.iterrows():
            term = str(row['word']).strip()
            definition = str(row['definition']).strip()
            
            if not term or not definition:
                continue
            
            normalized = self._normalize_term(term)
            if normalized not in term_defs:
                term_defs[normalized] = []
            term_defs[normalized].append(definition)
        
        # Keep shortest definition for each term
        for normalized_term, defs in term_defs.items():
            shortest_def = min(defs, key=len)
            self.glossary[normalized_term] = shortest_def
        
        # Build regex patterns sorted by term length (longest first)
        # This ensures longer multi-word terms match before shorter substrings
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            # Create word-boundary pattern for exact phrase matching
            # Escape special regex characters in the term
            escaped_term = re.escape(term)
            # Use word boundaries but allow partial matches for acronyms
            pattern = re.compile(r'\b' + escaped_term + r'\b', re.IGNORECASE)
            self.term_patterns.append((term, pattern))
        
        print(f"Loaded {len(self.glossary)} unique terms from {self.glossary_path}")

    def retrieve(
        self,
        sentence: str,
        max_definitions: int = 10
    ) -> List[Tuple[str, str]]:
        """
        Retrieve definitions for terms found in the sentence.

        Args:
            sentence: Input sentence to search for terms
            max_definitions: Maximum number of definitions to return

        Returns:
            List of (term, definition) tuples, ordered by first occurrence
        """
        if not sentence.strip():
            return []
        
        matched_terms: List[Tuple[str, int]] = []  # (term, position)
        matched_set = set()  # Track already matched terms
        
        for normalized_term, pattern in self.term_patterns:
            if normalized_term in matched_set:
                continue
            
            match = pattern.search(sentence)
            if match:
                matched_terms.append((normalized_term, match.start()))
                matched_set.add(normalized_term)
        
        # Sort by position in sentence, then take top max_definitions
        matched_terms.sort(key=lambda x: x[1])
        matched_terms = matched_terms[:max_definitions]
        
        # Return term-definition pairs
        return [(term, self.glossary[term]) for term, _ in matched_terms]

    def get_coverage_stats(
        self,
        sentences: List[str],
        max_definitions: int = 10
    ) -> Dict:
        """
        Compute coverage statistics over a list of sentences.

        Args:
            sentences: List of input sentences
            max_definitions: Maximum definitions per sentence

        Returns:
            Dictionary with coverage statistics
        """
        total_sentences = len(sentences)
        sentences_with_matches = 0
        total_matches = 0
        all_matched_terms: List[str] = []
        
        for sentence in sentences:
            matches = self.retrieve(sentence, max_definitions=max_definitions)
            if matches:
                sentences_with_matches += 1
                total_matches += len(matches)
                all_matched_terms.extend([term for term, _ in matches])
        
        # Count term frequencies
        from collections import Counter
        term_counts = Counter(all_matched_terms)
        
        return {
            'total_sentences': total_sentences,
            'sentences_with_matches': sentences_with_matches,
            'coverage_rate': sentences_with_matches / total_sentences if total_sentences > 0 else 0.0,
            'total_matches': total_matches,
            'avg_matches_per_sentence': total_matches / total_sentences if total_sentences > 0 else 0.0,
            'avg_matches_per_matched_sentence': total_matches / sentences_with_matches if sentences_with_matches > 0 else 0.0,
            'top_10_terms': term_counts.most_common(10),
            'unique_matched_terms': len(term_counts)
        }


if __name__ == "__main__":
    # Quick smoke test
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("word,definition\n")
        f.write("RCT,randomized controlled trial\n")
        f.write("CI,confidence interval\n")
        f.write("BMI,body mass index\n")
        temp_path = f.name
    
    retriever = GlossaryRetriever(temp_path)
    
    test_sentence = "We included 5 RCTs with 95% CI data and BMI measurements."
    matches = retriever.retrieve(test_sentence)
    
    print(f"\nTest sentence: {test_sentence}")
    print(f"\nMatched terms:")
    for term, definition in matches:
        print(f"  {term}: {definition}")
    
    import os
    os.unlink(temp_path)
