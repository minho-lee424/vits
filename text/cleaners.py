import re
from text.japanese import (
    japanese_to_romaji_with_accent,
    japanese_to_ipa,
    japanese_to_ipa2,
    japanese_to_ipa3,
)
from text.korean import (
    latin_to_hangul,
    number_to_hangul,
    divide_hangul,
    korean_to_lazy_ipa,
    korean_to_ipa,
)


def japanese_cleaners(text):
    text = japanese_to_romaji_with_accent(text)
    text = re.sub(r"([A-Za-z])$", r"\1.", text)
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace("ts", "ʦ").replace("...", "…")


def korean_cleaners(text):
    """Pipeline for Korean text"""
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = divide_hangul(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    return text
