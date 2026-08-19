"""
Block 1: extract candidate object noun phrases from a language instruction.

Each mentioned object yields an ordered list of candidate strings to try
against SAM3, from most specific to most generic (e.g. "black bowl" -> "bowl").
This exists because LIBERO object catalog names (e.g. "black bowl") sometimes
don't match the object's actual rendered appearance (e.g. it's silver), which
makes SAM3's zero-shot text grounding reject the full phrase outright even
though the plain head noun grounds it correctly. See cog-xvla/data QA notes.

No training involved -- this wraps spaCy's noun-chunk parser.
"""

from __future__ import annotations

from typing import List

import spacy

_NLP = None

# Words that are grammatically noun chunks but never refer to a graspable
# object in LIBERO-style instructions.
_STOP_HEADS = {"it", "them", "this", "that", "there"}

# Leading tokens to strip from a noun chunk before treating it as an
# object reference (determiners, possessives).
_STRIP_LEADING_POS = {"DET", "PRON"}


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _clean_chunk(chunk) -> List[str]:
    """Return a token list for a noun chunk with leading determiners/prons stripped."""
    tokens = [t for t in chunk]
    while tokens and tokens[0].pos_ in _STRIP_LEADING_POS:
        tokens = tokens[1:]
    return tokens


def extract_object_phrases(instruction: str) -> List[List[str]]:
    """
    Args:
        instruction: a LIBERO-style language instruction, e.g.
            "pick up the black bowl between the plate and the ramekin
             and place it on the plate"

    Returns:
        A list of candidate lists, one per distinct object mention, e.g.:
        [["black bowl", "bowl"], ["plate"], ["ramekin"], ["plate"]]
        Candidates are ordered most-specific -> most-generic; a downstream
        SAM3 wrapper should try them in order and stop at the first hit.
    """
    nlp = _get_nlp()
    doc = nlp(instruction)

    results: List[List[str]] = []
    for chunk in doc.noun_chunks:
        tokens = _clean_chunk(chunk)
        if not tokens:
            continue
        text = " ".join(t.text for t in tokens).lower().strip()
        if not text or text in _STOP_HEADS:
            continue

        candidates = [text]
        # Head-noun fallback: drop leading adjectives/compound-noun modifiers,
        # keep the syntactic head (typically the last token, e.g. "bowl" in
        # "black bowl"). Only add if it differs from the full phrase and is
        # itself a plausible object noun (not an adjective etc.).
        head = chunk.root
        head_text = head.text.lower()
        if head_text != text and head.pos_ in ("NOUN", "PROPN"):
            candidates.append(head_text)

        results.append(candidates)

    return results


if __name__ == "__main__":
    examples = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "open the middle drawer of the cabinet",
        "turn on the stove and put the moka pot on it",
        "pick up the alphabet soup and place it in the basket",
    ]
    for ex in examples:
        print(ex)
        for cands in extract_object_phrases(ex):
            print("  ", cands)
