from typing import List, Tuple, Callable, Any, Optional


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


def find_overlap_chunks(_text_chunks: List[Any], convert_any_to_str: Optional[Callable[..., str]] = None) -> List[Tuple[int]]:
    """
    Find overlaps over consecutive texts

    Args:
        _text_chunks (List[str]): List of strings to be tested, already ordered.
        convert_any_to_str (Optional[Callable[..., str]]): convert of an element of _text_chunks to str

    Returns:
        List[Tuple[int]]: List of output from find_overlap for each consecutive strings
    """
    # Convert the elements of _text_chunks to string if needed
    if type(_text_chunks[0]) != str:
        if convert_any_to_str is None:
            raise ValueError
        _text_chunks_str = [convert_any_to_str(chunk) for chunk in _text_chunks]
    else:
        _text_chunks_str = _text_chunks
    # Calculate the overlap between each consecutive chunks
    nb_chunks = len(_text_chunks_str)
    return [
        find_overlap(_text_chunks_str[ii], _text_chunks_str[ii + 1])
        for ii in range(nb_chunks - 1)
    ]
