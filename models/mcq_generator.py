# models/mcq_generator.py
import random
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.text_cleaner import clean_text
from utils.difficulty_classifier import classify_difficulty

# Load once on import
_sbert_model = SentenceTransformer('all-mpnet-base-v2')   # embeddings
_kw_model = KeyBERT(model=_sbert_model)
# T5-small for generation
_t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
_t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def _prepare_prompt_for_t5(context, difficulty, num_questions):
    # create a concise prompt instructing T5 to output JSON-like MCQs
    prompt = f"Generate {num_questions} multiple choice questions from the text below. Difficulty: {difficulty}. For each question output question ||| option1 ||| option2 ||| option3 ||| option4 ||| correct_index(1-4)\n\nText:\n{context}\n\nFormat strictly: one QA per line, fields separated by |||"
    return prompt

def _parse_t5_output(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    mcqs = []
    for line in lines:
        parts = [p.strip() for p in line.split("|||")]
        if len(parts) >= 6:
            q = parts[0]
            opts = parts[1:5]
            idx = int(parts[5]) - 1
            labeled = list(zip(['A','B','C','D'], opts))
            mcqs.append({
                "question": q,
                "options": labeled,
                "answer_text": opts[idx],
                "answer_index": idx
            })
    return mcqs

def generate_mcqs(text, num_questions=5, difficulty="medium"):
    text = clean_text(text)
    if not text:
        return []
    # extract keywords to help generate questions
    keywords = _kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), top_n=15)
    # pick context sentences: sample top N sentences using SBERT similarity
    sents = text.split(".")
    sents = [s.strip() for s in sents if len(s.strip())>20]
    # construct context chunk (limit tokens)
    context = " ".join(sents[:50])
    # prepare prompt for T5
    prompt = _prepare_prompt_for_t5(context, difficulty, num_questions)
    input_ids = _t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    outputs = _t5_model.generate(input_ids, max_length=512, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50)
    text_out = _t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    mcqs = _parse_t5_output(text_out)
    # fallback to simpler generator if T5 fails or returns <=0
    if not mcqs:
        mcqs = []
        # simple heuristic: use sentences and create blanks at keywords
        for s in sents[:num_questions*3]:
            if len(mcqs) >= num_questions:
                break
            # pick candidate token (noun-like) by splitting
            words = [w for w in s.split() if w.isalpha() and len(w)>3]
            if not words:
                continue
            answer = random.choice(words)
            opts = [answer]
            # create distractors randomly from keywords
            kd = [k[0] for k in keywords if k[0].lower()!=answer.lower()]
            random.shuffle(kd)
            for k in kd[:3]:
                opts.append(k)
            while len(opts) < 4:
                opts.append("None")
            random.shuffle(opts)
            labeled = list(zip(['A','B','C','D'], opts))
            mcqs.append({"question": s.replace(answer, "_____"), "options": labeled, "answer_text": answer})
    # ensure exactly num_questions
    return mcqs[:num_questions]
