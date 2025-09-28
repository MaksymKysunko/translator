import os
import math
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from blingfire import text_to_sentences as split_sentences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import logging


# ====== НАЛАШТУВАННЯ ======
INPUT_PATH = "book_ru.txt"            # вхідний TXT (UTF-8)
OUTPUT_PATH = "book_ua.txt"           # куди зберегти переклад
MODEL_ID = "facebook/nllb-200-distilled-1.3B"  # можна 600M або 3.3B
SRC_LANG = "rus_Cyrl"                 # російська
TGT_LANG = "ukr_Cyrl"                 # англійська

# Розмір батчу та довжини. Якщо б'є в OOM — зменш.
BATCH_SIZE = 16
MAX_SRC_TOKENS = 512                  # максимум для вхідних токенів
MAX_NEW_TOKENS = 256                  # максимум згенерованих токенів

# Параметри генерації: якість/стабільність
GEN_KW = dict(
    num_beams=4,                      # трохи підвищує якість
    length_penalty=1.0,
    no_repeat_ngram_size=3
)
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, 
                                                  dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                                                  #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tok, model, device


def read_paragraphs(path: str) -> List[str]:
    text = Path(path).read_text(encoding="utf-8")
    # Розбиваємо за подвійними/одинарними переносами — збережемо структуру
    # Далі всередині кожного абзацу — розіб'ємо на речення для кращої якості.
    raw_pars = [p.strip() for p in text.split("\n")]
    # Прибираємо порожні
    return [p for p in raw_pars if p]


def chunk_sentences(paragraph: str, max_chars: int = 1200) -> List[str]:
    """
    Розбиваємо абзац на речення і пакуємо в шматки ~max_chars,
    щоб не перевищити контекст моделі.
    """
    sents = split_sentences(paragraph).split("\n")
    chunks, cur = [], ""
    for s in sents:
        s = s.strip()
        if s:
            chunks.append(s)
 
    return chunks


def translate_chunks(chunks: List[str], tok, model, device: str) -> List[str]:
    tok.src_lang = SRC_LANG
    outputs = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Translating"):
        batch = chunks[i:i + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SRC_TOKENS).to(device)
        gen_ids = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(TGT_LANG),
            max_new_tokens=MAX_NEW_TOKENS,
            **GEN_KW
        )
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend(texts)

    return outputs

def assert_equal_sentence_count(src: str, tgt: str):
    src_sents = [s.strip() for s in split_sentences(src).split("\n") if s.strip()]
    tgt_sents = [s.strip() for s in split_sentences(tgt).split("\n") if s.strip()]
    if len(src_sents) != len(tgt_sents):
        logging.warning(
            f"Sentence count mismatch: src={len(src_sents)}, "
            f"tgt={len(tgt_sents)} | src_head='{src[:60]}...' "
            f"| tgt_head='{tgt[:60]}...'"
        )

def main():
    assert Path(INPUT_PATH).exists(), f"No input file: {INPUT_PATH}"
    tok, model, device = load_model(MODEL_ID)

    paragraphs = read_paragraphs(INPUT_PATH)
    out_lines = []

    for p in tqdm(paragraphs, desc="Paragraphs"):
        # 1) розбиваємо абзац на керовані шматки за реченнями
        parts = chunk_sentences(p, max_chars=1200)
        # 2) перекладаємо шматками
        translated_parts = translate_chunks(parts, tok, model, device)
        # 3) склеюємо назад, зберігаючи абзац
        for original, translated in zip(parts, translated_parts):
            assert_equal_sentence_count(original, translated)
        out_lines.append("\n".join(translated_parts))

    Path(OUTPUT_PATH).write_text("\n\n".join(out_lines), encoding="utf-8")
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
