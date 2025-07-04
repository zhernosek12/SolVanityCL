def parse_wallet_pattern(pattern: str):
    wallet_start = ""
    wallet_end = ""

    if "__" in pattern:
        wallet_start, wallet_end = pattern.split("__", 1)
    elif pattern.endswith("_"):
        wallet_start = pattern[:-1]
    elif pattern.startswith("_"):
        wallet_end = pattern[1:]

    return wallet_start, wallet_end
