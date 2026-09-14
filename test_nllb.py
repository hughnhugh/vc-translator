import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "facebook/nllb-200-distilled-600M"

print("Loading NLLB tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to("cuda").half()

tokenizer.src_lang = "eng_Latn"
text = "Why are you laughing at me?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
forced_bos_token_id = tokenizer.convert_tokens_to_ids("azj_Latn")
output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=64)
result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

print(f"EN: {text}")
print(f"AZ: {result}")
