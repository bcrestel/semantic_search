from typing import List, Tuple


def find_overlap(text_0, text_1, max_overlap: int = 100) -> Tuple[int]:
    """
    Find overlap between two texts and calculate length of overlap

    Args:
        text_0 (str): First text.
        text_1 (str): Second text.
        max_overlap (int, optional): Maximum length of text to detect overlap. Defaults to 100.

    Returns:
        Tuple[int]: length of overlap (0 if no overlap), index of start of overlap in text_0
    """
    # Find start of overlap
    chunk_detect_overlap = text_1.split(".")[0][:max_overlap]
    idx_start_overlap = text_0.find(chunk_detect_overlap)
    if idx_start_overlap < 0:
        return 0, idx_start_overlap
    else:
        return len(text_0) - idx_start_overlap, idx_start_overlap


def find_overlap_chunks(_text_chunks: List[str]) -> List[Tuple[int]]:
    """
    Find overlaps over consecutive texts

    Args:
        _text_chunks (List[str]): List of strings to be tested, already ordered.

    Returns:
        List[Tuple[int]]: List of output from find_overlap for each consecutive strings
    """
    nb_chunks = len(_text_chunks)
    return [
        find_overlap(_text_chunks[ii], _text_chunks[ii + 1])
        for ii in range(nb_chunks - 1)
    ]
