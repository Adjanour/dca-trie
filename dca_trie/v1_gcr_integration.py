"""
Integration: DCA-Trie v1 into GCR's prediction pipeline.

Two approaches:
  1. V1PromptBuilder: subclass of GraphConstrainedPromptBuilder that
     overrides get_graph_index() to use V1TrieBuilder.
  2. patch_prompt_builder(): monkey-patch process_input for quick swap
     without changing GCR imports.

Usage (approach 1 - subclass):
    from dca_trie.v1_gcr_integration import V1PromptBuilder
    input_builder = V1PromptBuilder(tokenizer, scorer=scorer, tau=0.4)
    trie = input_builder.get_graph_index(question_dict)

Usage (approach 2 - monkey-patch):
    from dca_trie.v1_gcr_integration import patch_prompt_builder
    patch_prompt_builder(tokenizer, scorer, tau=0.4)
    # Now all GraphConstrainedPromptBuilder instances use V1TrieBuilder
"""

from typing import Optional
from gcr.src.qa_prompt_builder import GraphConstrainedPromptBuilder
from dca_trie.semantic_scorer import SemanticScorer
from dca_trie.v1_trie_builder import V1TrieBuilder


class V1PromptBuilder(GraphConstrainedPromptBuilder):
    """
    Drop-in replacement for GraphConstrainedPromptBuilder.

    Overrides get_graph_index() to use V1TrieBuilder with semantic filtering.

    Usage:
        input_builder = V1PromptBuilder(
            tokenizer, scorer=scorer, tau=0.4,
            prompt="zero-shot", undirected=False, index_path_length=2,
        )
        trie = input_builder.get_graph_index(question_dict)
    """

    def __init__(
        self,
        tokenizer,
        scorer: SemanticScorer,
        tau: float = 0.3,
        prompt="zero-shot",
        undirected=False,
        index_path_length=2,
        add_rule=False,
        path_to_str_fn: Optional[callable] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            prompt=prompt,
            undirected=undirected,
            index_path_length=index_path_length,
            add_rule=add_rule,
        )
        self._v1_builder = V1TrieBuilder(
            tokenizer=tokenizer,
            scorer=scorer,
            tau=tau,
            index_path_length=index_path_length,
            undirected=undirected,
            path_to_str_fn=path_to_str_fn,
        )

    def get_graph_index(self, question_dict):
        return self._v1_builder.build_filtered_trie(question_dict)

    def get_graph_index_with_scores(self, question_dict):
        return self._v1_builder.build_filtered_trie_with_scores(question_dict)


# Keep a reference to the original for restore
_original_process_input = GraphConstrainedPromptBuilder.process_input


def _patched_process_input(self, question_dict, return_trie=True):
    """
    Patched version of GraphConstrainedPromptBuilder.process_input.
    Delegates get_graph_index to V1PromptBuilder.
    """
    if return_trie:
        # Use V1PromptBuilder.get_graph_index via the v1_builder
        # stored on the instance by the patch
        trie = self._v1_builder.build_filtered_trie(question_dict)
    else:
        trie = None

    from gcr.src.utils import build_graph, get_truth_paths, path_to_string
    question = question_dict["question"]
    start_node = question_dict["q_entity"]
    answer_node = question_dict["a_entity"]
    choices = question_dict.get("choices", [])

    g = build_graph(question_dict["graph"], self.undirected)
    truth_paths = get_truth_paths(start_node, answer_node, g)
    ground_paths = [path_to_string(path) for path in truth_paths]

    start_entities = question_dict.get("q_entity", [])
    input_text = self.format_input_with_template(question, start_entities, choices)

    return trie, input_text, ground_paths


def patch_prompt_builder(tokenizer, scorer, tau=0.3,
                          path_to_str_fn: Optional[callable] = None):
    """
    Monkey-patch GraphConstrainedPromptBuilder.process_input to use V1TrieBuilder.

    After calling this, ALL instances of GraphConstrainedPromptBuilder will
    use DCA-Trie v1's semantically filtered trie.

    To restore:
        from dca_trie.v1_gcr_integration import restore_prompt_builder
        restore_prompt_builder()
    """
    v1_builder = V1TrieBuilder(
        tokenizer=tokenizer,
        scorer=scorer,
        tau=tau,
        path_to_str_fn=path_to_str_fn,
    )

    # Attach builder and override process_input
    def _make_patched(v1_builder):
        def _process_input(self, question_dict, return_trie=True):
            if return_trie:
                trie = v1_builder.build_filtered_trie(question_dict)
            else:
                trie = None

            from gcr.src.utils import build_graph, get_truth_paths, path_to_string
            question = question_dict["question"]
            start_node = question_dict["q_entity"]
            answer_node = question_dict["a_entity"]
            choices = question_dict.get("choices", [])

            g = build_graph(question_dict["graph"], self.undirected)
            truth_paths = get_truth_paths(start_node, answer_node, g)
            ground_paths = [path_to_string(path) for path in truth_paths]

            start_entities = question_dict.get("q_entity", [])
            input_text = self.format_input_with_template(question, start_entities, choices)

            return trie, input_text, ground_paths

        return _process_input

    GraphConstrainedPromptBuilder.process_input = _make_patched(v1_builder)


def restore_prompt_builder():
    """Restore GCR's original process_input after patching."""
    GraphConstrainedPromptBuilder.process_input = _original_process_input
