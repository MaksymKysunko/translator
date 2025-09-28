# save as translate_ru_to_ukr_preserve_lines.py
from pathlib import Path
from typing import List, Dict
import re
import logging

import torch
from tqdm import tqdm
from blingfire import text_to_sentences as split_sentences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===== Settings =====
INPUT_PATH = "book_ru.txt"
OUTPUT_PATH = "book_ukr.txt"

#MODEL_ID = "facebook/nllb-200-distilled-600M"  # легше для RTX 3050; можна 1.3B якщо вистачає VRAM
MODEL_ID = "facebook/nllb-200-distilled-1.3B"  # можна 600M або 3.3B
SRC_LANG = "rus_Cyrl"
#TGT_LANG = "eng_Latn" 
TGT_LANG = "ukr_Cyrl" 


BATCH_SIZE = 16
MAX_SRC_TOKENS = 512
MAX_NEW_TOKENS = 256

NUM_BEAMS = 2             # 1..4: 1 швидше/стабільніше; 2-4 трохи краща якість
NO_REPEAT_NGRAM_SIZE = 3   # зменшує повтори
# =====================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    return tok, model


def split_lines_with_ends(text: str) -> List[str]:
    """
    Повертає список рядків з оригінальними закінченнями: splitlines(keepends=True).
    Приклад: 'A\\n\\nB' -> ['A\\n', '\\n', 'B']
    """
    return text.splitlines(keepends=True)


def line_to_sentences_with_prefix(line: str) -> Dict:
    """
    Витягує контент рядка, його prefix (пробіли/тире на початку) і суфікс (оригінальні \n/\r\n),
    розбиває контент на речення.
    Повертає: {type, raw? | prefix, suffix, sents}
    """
    # Якщо рядок складається лише з переводу каретки — просто повертаємо як blank
    if line.strip("\r\n") == "":
        return {"type": "blank", "raw": line}

    # Відрізаємо кінцеві \n/\r\n, зберігаємо їх як suffix
    content = line.rstrip("\r\n")
    suffix = line[len(content):]  # те, що відрізали (може бути "" або "\n"/"\r\n")

    # Префікс: пробіли + опц. тире діалогу (—/–/-) + пробіл
    m = re.match(r'^(\s*[—–-]?\s*)', content)
    prefix = m.group(1) if m else ""
    core = content[len(prefix):]

    sents = [s.strip() for s in split_sentences(core).split("\n") if s.strip()]
    return {"type": "line", "prefix": prefix, "suffix": suffix, "sents": sents}


def build_queue(lines: List[str]) -> (List[Dict], List[str]):
    """
    Готує метадані для реконструкції і глобальну чергу речень для перекладу.
    meta[i] описує i-й елемент оригінального тексту (рядок або порожній рядок).
    queue — усі речення у вихідному порядку.
    """
    meta, queue = [], []
    for line in lines:
        item = line_to_sentences_with_prefix(line)
        meta.append(item)
        if item["type"] == "line":
            queue.extend(item["sents"])
    return meta, queue


def translate_queue(queue: List[str], tok, model) -> List[str]:
    tok.src_lang = SRC_LANG
    outputs: List[str] = []
    for i in tqdm(range(0, len(queue), BATCH_SIZE), desc="Translating"):
        batch = queue[i:i + BATCH_SIZE]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SRC_TOKENS
        ).to(model.device)
        with torch.inference_mode():
            gen_ids = model.generate(
                **enc,
                forced_bos_token_id=tok.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outputs.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
    return outputs


def reconstruct(meta: List[Dict], translations: List[str]) -> str:
    """
    Відновлює документ 1-в-1 за оригінальними рядками:
    для кожного 'line' бере стільки перекладених речень, скільки було у sents,
    і склеює їх назад як "prefix + ' '.join(tgt_sents) + suffix".
    Порожні рядки залишаються як є.
    """
    out_parts: List[str] = []
    idx = 0
    for item in meta:
        if item["type"] == "blank":
            out_parts.append(item["raw"])
            continue
        cnt = len(item["sents"])
        tgt_slice = translations[idx: idx + cnt]
        idx += cnt
        out_parts.append(item["prefix"] + " ".join(tgt_slice).strip() + item["suffix"])
    # sanity check: весь переклад має бути використаний
    if idx != len(translations):
        logging.warning(f"Unused translations: used={idx}, total={len(translations)}")
    return "".join(out_parts)


def main():
    assert Path(INPUT_PATH).exists(), f"No input file: {INPUT_PATH}"
    raw = Path(INPUT_PATH).read_text(encoding="utf-8")

    # 1) Розбиваємо на рядки з оригінальними \n/\r\n
    lines = split_lines_with_ends(raw)

    # 2) Будуємо мету і глобальну чергу речень
    meta, queue = build_queue(lines)
    logging.info(f"lines={len(lines)}, sentences={len(queue)}")

    # 3) Переклад
    tok, model = load_model()
    translations = translate_queue(queue, tok, model)

    if len(translations) != len(queue):
        logging.warning(f"mismatch: in={len(queue)} out={len(translations)}")

    # 4) Реконструкція з тим самим форматуванням рядків
    result = reconstruct(meta, translations)
    Path(OUTPUT_PATH).write_text(result, encoding="utf-8")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
