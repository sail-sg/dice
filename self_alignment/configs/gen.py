DEFAULT = {
    "temperature": 0.7,
    "top_p": 0.9,
}

LLAMA3 = {
    "temperature": 0.9,
    "top_p": 1.0,
    "stop_token_ids": [128001, 128009]
}


GEN_CONFIG_MAP = {
    "default": DEFAULT,
    "llama3": LLAMA3,
}