"""Few-shot example retrieval for prompt-based simplification."""

from typing import List, Tuple
import numpy as np
from collections import Counter


class FewShotRetriever:
    """Retrieves similar training examples for few-shot prompting."""

    def __init__(
        self,
        train_complex: List[str],
        train_simple: List[List[str]],
        method: str = "lexical"
    ):
        """
        Initialize the few-shot retriever.

        Args:
            train_complex: List of complex training sentences
            train_simple: List of lists of simple reference sentences
            method: Retrieval method ('lexical' for word overlap, 'random' for random sampling)
        """
        self.train_complex = train_complex
        self.train_simple = train_simple
        self.method = method
        
        if method == "lexical":
            # Precompute word sets for each training example
            self.train_word_sets = [
                set(sent.lower().split()) for sent in train_complex
            ]
    
    def _compute_lexical_similarity(self, query: str, train_idx: int) -> float:
        """Compute Jaccard similarity between query and training example."""
        query_words = set(query.lower().split())
        train_words = self.train_word_sets[train_idx]
        
        if not query_words or not train_words:
            return 0.0
        
        intersection = len(query_words & train_words)
        union = len(query_words | train_words)
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve(
        self,
        query_sentence: str,
        k: int = 3,
        exclude_self: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Retrieve k most similar training examples.

        Args:
            query_sentence: The sentence to find examples for
            k: Number of examples to retrieve
            exclude_self: Whether to exclude exact matches (useful if query is from train)

        Returns:
            List of (complex, simple) example pairs
        """
        if self.method == "random":
            # Random sampling
            indices = np.random.choice(len(self.train_complex), size=min(k, len(self.train_complex)), replace=False)
            examples = []
            for idx in indices:
                complex_ex = self.train_complex[idx]
                simple_ex = self.train_simple[idx][0] if self.train_simple[idx] else complex_ex
                examples.append((complex_ex, simple_ex))
            return examples
        
        elif self.method == "lexical":
            # Compute similarities
            similarities = []
            for idx in range(len(self.train_complex)):
                if exclude_self and self.train_complex[idx] == query_sentence:
                    continue
                sim = self._compute_lexical_similarity(query_sentence, idx)
                similarities.append((sim, idx))
            
            # Sort by similarity and take top k
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_k = similarities[:k]
            
            # Build example pairs
            examples = []
            for sim, idx in top_k:
                complex_ex = self.train_complex[idx]
                simple_ex = self.train_simple[idx][0] if self.train_simple[idx] else complex_ex
                examples.append((complex_ex, simple_ex))
            
            return examples
        
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")


if __name__ == "__main__":
    # Quick smoke test
    train_complex = [
        "We included 5 RCTs with 1000 participants.",
        "The intervention reduced mortality significantly (p < 0.01).",
        "Participants received daily medication for 6 months."
    ]
    
    train_simple = [
        ["We found 5 studies with 1000 people."],
        ["The treatment reduced deaths significantly."],
        ["Participants took daily medication for 6 months."]
    ]
    
    retriever = FewShotRetriever(train_complex, train_simple, method="lexical")
    
    test_query = "We analyzed 10 RCTs with 2000 participants receiving treatment."
    examples = retriever.retrieve(test_query, k=2)
    
    print(f"Query: {test_query}\n")
    print("Retrieved examples:")
    for i, (complex_ex, simple_ex) in enumerate(examples, 1):
        print(f"\n{i}. Complex: {complex_ex}")
        print(f"   Simple: {simple_ex}")
