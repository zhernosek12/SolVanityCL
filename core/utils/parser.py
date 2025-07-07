def parse_wallet_pattern(token_address: str, pattern: str):
    wallet_start = ""
    wallet_end = ""

    if "__" in pattern:
        start_part, end_part = pattern.split("__", 1)
        if start_part.isdigit():
            wallet_start = token_address[:int(start_part)]
        else:
            wallet_start = start_part
        if end_part.isdigit():
            wallet_end = token_address[-int(end_part):]
        else:
            wallet_end = end_part
    elif pattern.endswith("_"):
        start_part = pattern[:-1]
        if start_part.isdigit():
            wallet_start = token_address[:int(start_part)]
        else:
            wallet_start = start_part
    elif pattern.startswith("_"):
        end_part = pattern[1:]
        if end_part.isdigit():
            wallet_end = token_address[-int(end_part):]
        else:
            wallet_end = end_part
    else:
        wallet_start = pattern

    return wallet_start, wallet_end
