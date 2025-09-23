def deep_merge(d1: dict, d2: dict) -> dict:
    """递归合并两个字典，d2 覆盖 d1"""
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1
