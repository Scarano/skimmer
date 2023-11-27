

def abbrev(s: str, max_len: int=20) -> str:
    if len(s) <= max_len:
        return s
    else:
        return s[:max_len//2] + '\u2026' + s[-max_len//2+1:]