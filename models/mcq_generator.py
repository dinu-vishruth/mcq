# models/mcq_generator.py
"""
MCQ Generator — uses fine-tuned Flan-T5 (GPU) when available,
falls back to NLTK NLP pipeline (CPU) otherwise.

After training (python train/train_model.py), the fine-tuned model
will be automatically detected and used for much better question quality.
"""

import os
import re
import random
import math
from collections import Counter

import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize, ne_chunk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tree import Tree

from utils.text_cleaner import clean_text

# ---- NLTK resources ----
nltk_dir = "/tmp/nltk_data" if os.environ.get("VERCEL") else nltk.data.path[0]
if os.environ.get("VERCEL"):
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)

for _res in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "wordnet",
             "omw-1.4", "stopwords"]:
    nltk.download(_res, download_dir=nltk_dir, quiet=True)

_STOPWORDS = set(stopwords.words("english"))

# ---- Try to load fine-tuned T5 model ----
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flan-t5-mcq")
_t5_model = None
_t5_tokenizer = None
_t5_device = None

def _load_t5_model():
    """Try to load the fine-tuned Flan-T5 model. Returns True if successful."""
    global _t5_model, _t5_tokenizer, _t5_device

    if not os.path.exists(_MODEL_DIR):
        return False

    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        _t5_tokenizer = T5Tokenizer.from_pretrained(_MODEL_DIR, legacy=True)
        _t5_model = T5ForConditionalGeneration.from_pretrained(_MODEL_DIR)

        if torch.cuda.is_available():
            _t5_device = torch.device("cuda")
            _t5_model.to(_t5_device)
            gpu = torch.cuda.get_device_name(0)
            print(f"  [GPU] MCQ Model loaded on GPU: {gpu}")
        else:
            _t5_device = torch.device("cpu")
            _t5_model.to(_t5_device)
            print(f"  [CPU] MCQ Model loaded on CPU")

        _t5_model.eval()
        return True
    except Exception as e:
        print(f"  [WARNING] Could not load T5 model: {e}")
        return False


# Attempt to load on import
_USE_T5 = _load_t5_model()
if _USE_T5:
    print("  [SUCCESS] Using fine-tuned Flan-T5 for MCQ generation")
else:
    print("  [INFO] Using NLTK NLP pipeline (train a model for better quality: python train/train_model.py)")


# =================================================================
#  T5-BASED GENERATION (when model is available)
# =================================================================

def _t5_generate_mcqs(text, num_questions=5, difficulty="medium"):
    """Generate MCQs using fine-tuned Flan-T5."""
    import torch

    difficulty_word = {"easy": "easy", "medium": "medium", "hard": "hard"}.get(difficulty, "medium")
    sentences = sent_tokenize(text)

    # Select the best chunks of text to generate questions from
    chunks = _get_text_chunks(text, max_chunk_len=400)
    mcqs = []
    used_questions = set()

    for chunk in chunks:
        if len(mcqs) >= num_questions:
            break

        prompt = f"Generate a {difficulty_word} multiple choice question from this text:\n\n{chunk}"

        input_enc = _t5_tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(_t5_device)

        with torch.no_grad():
            outputs = _t5_model.generate(
                **input_enc,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
            )

        output_text = _t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        mcq = _parse_t5_output(output_text, chunk)

        if mcq and mcq["question"] not in used_questions:
            used_questions.add(mcq["question"])
            mcqs.append(mcq)

    return mcqs[:num_questions]


def _get_text_chunks(text, max_chunk_len=400):
    """Split text into meaningful chunks for T5 input."""
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_chunk_len:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _parse_t5_output(output, source=""):
    """Parse T5 model output into structured MCQ dict."""
    try:
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]

        question = ""
        options = []
        answer_label = ""
        explanation = ""

        for line in lines:
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
            elif re.match(r'^[A-D]\)', line):
                label = line[0]
                text = line[3:].strip()
                options.append({"label": label, "text": text})
            elif line.startswith("Answer:"):
                answer_label = line[len("Answer:"):].strip()
            elif line.startswith("Explanation:"):
                explanation = line[len("Explanation:"):].strip()

        if not question or len(options) < 4 or not answer_label:
            return None

        answer_text = next(
            (o["text"] for o in options if o["label"] == answer_label), options[0]["text"]
        )

        if not explanation:
            explanation = f"The correct answer is '{answer_text}'."

        return {
            "question": question,
            "options": options,
            "answer_text": answer_text,
            "answer_label": answer_label,
            "explanation": explanation,
        }
    except Exception:
        return None


# =================================================================
#  NLTK NLP FALLBACK (when no trained model)
# =================================================================

def _get_sentences(text):
    """Extract meaningful sentences."""
    raw_sents = sent_tokenize(text)
    good = []
    for s in raw_sents:
        s = s.strip()
        words = s.split()
        if len(words) < 6 or len(s) < 25:
            continue
        if s.startswith(("http", "www.", "©", "Page ")):
            continue
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.5:
            continue
        good.append(s)
    return good


def _tfidf_rank(sentences, full_text):
    """Rank sentences by TF-IDF importance."""
    all_tokens = word_tokenize(full_text.lower())
    all_tokens = [t for t in all_tokens if t.isalpha() and t not in _STOPWORDS and len(t) > 2]
    doc_tf = Counter(all_tokens)
    total = sum(doc_tf.values()) or 1

    scored = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        content = [t for t in tokens if t.isalpha() and t not in _STOPWORDS and len(t) > 2]
        score = sum(
            (doc_tf.get(t, 0) / total) * math.log(total / (1 + doc_tf.get(t, 0)))
            for t in content
        )
        try:
            tagged = pos_tag(word_tokenize(sent))
            tree = ne_chunk(tagged, binary=False)
            score += sum(1 for st in tree if isinstance(st, Tree)) * 0.3
        except Exception:
            pass
        if content:
            score /= len(content)
        scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


def _extract_entities(text):
    entities = {}
    try:
        for sent in sent_tokenize(text[:15000]):
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            tree = ne_chunk(tagged, binary=False)
            for subtree in tree:
                if isinstance(subtree, Tree):
                    ent = " ".join(w for w, _ in subtree.leaves())
                    if len(ent) > 1 and ent.lower() not in _STOPWORDS and not ent.isdigit():
                        entities[ent] = subtree.label()
    except Exception:
        pass
    return entities


def _extract_nouns(sentence):
    try:
        tagged = pos_tag(word_tokenize(sentence))
        nouns = [w for w, t in tagged
                 if t.startswith("NN") and len(w) > 3
                 and w.lower() not in _STOPWORDS and w.isalpha()]
        # Noun phrases
        for i in range(len(tagged) - 1):
            w1, t1 = tagged[i]
            w2, t2 = tagged[i + 1]
            if t1.startswith("NN") and t2.startswith("NN") and w1.isalpha() and w2.isalpha():
                nouns.append(f"{w1} {w2}")
        return nouns
    except Exception:
        return []


def _wordnet_distractors(word, n=6):
    distractors = set()
    for syn in wn.synsets(word.lower().replace(" ", "_"))[:3]:
        for hyp in syn.hypernyms():
            for lem in hyp.lemmas():
                name = lem.name().replace("_", " ")
                if name.lower() != word.lower():
                    distractors.add(name)
        for hyp in syn.hypernyms():
            for sib in hyp.hyponyms():
                for lem in sib.lemmas():
                    name = lem.name().replace("_", " ")
                    if name.lower() != word.lower():
                        distractors.add(name)
    return list(distractors)[:n]


def _context_distractors(answer, entities, all_nouns, n=6):
    answer_lower = answer.lower()
    label = entities.get(answer)
    dists = []
    if label:
        dists += [e for e, l in entities.items() if l == label and e.lower() != answer_lower]
    dists += [n_ for n_ in all_nouns
              if n_.lower() != answer_lower and n_ not in dists and len(n_) > 2]
    return dists[:n]


def _build_mcq(question, answer, distractors, source=""):
    opts = [answer] + distractors[:3]
    random.shuffle(opts)
    labeled = [{"label": l, "text": t} for l, t in zip("ABCD", opts)]
    ans_label = next(o["label"] for o in labeled if o["text"] == answer)
    short_src = (source[:200] + "...") if len(source) > 200 else source
    return {
        "question": question,
        "options": labeled,
        "answer_text": answer,
        "answer_label": ans_label,
        "explanation": f"The correct answer is '{answer}'. Based on: \"{short_src}\"",
    }


def _nltk_generate(text, num_questions, difficulty):
    """Generate MCQs using NLTK NLP pipeline (no model needed)."""
    sentences = _get_sentences(text)
    ranked = _tfidf_rank(sentences, text)
    entities = _extract_entities(text)
    all_nouns = list(set(
        n for s in sentences[:50] for n in _extract_nouns(s)
    ))
    random.shuffle(all_nouns)

    if not ranked:
        return []

    templates = {
        "easy": "Fill in the blank: {}",
        "medium": random.choice([
            "Which of the following correctly completes: {}",
            "According to the passage: {}",
            "Based on the text, fill in the blank: {}",
        ]),
        "hard": random.choice([
            "Analyze and identify the correct term: {}",
            "Which option best completes this statement: {}",
            "Critically evaluate: {}",
        ]),
    }

    mcqs = []
    used = set()

    for sent in ranked:
        if len(mcqs) >= num_questions:
            break

        # Get candidates (entities + nouns)
        candidates = []
        try:
            tagged = pos_tag(word_tokenize(sent))
            tree = ne_chunk(tagged, binary=False)
            for st in tree:
                if isinstance(st, Tree):
                    ent = " ".join(w for w, _ in st.leaves())
                    if len(ent) > 1 and ent.lower() not in used:
                        candidates.append(ent)
        except Exception:
            pass

        candidates += [n for n in _extract_nouns(sent) if n.lower() not in used]

        if not candidates:
            continue

        answer = max(candidates, key=len)
        used.add(answer.lower())

        template = templates.get(difficulty, templates["medium"])
        blanked = sent.replace(answer, "______")
        question = template.format(blanked) if isinstance(template, str) else template.format(blanked)

        dists = _context_distractors(answer, entities, all_nouns)
        dists += _wordnet_distractors(answer)
        dists = list(dict.fromkeys(
            d for d in dists if d.lower() != answer.lower() and d.lower() not in used
        ))

        if difficulty == "hard":
            dists.sort(key=lambda d: abs(len(d) - len(answer)))

        dists = dists[:3]
        while len(dists) < 3:
            dists.append("None of the above")

        mcqs.append(_build_mcq(question, answer, dists[:3], sent))

    return mcqs


# =================================================================
#  MAIN ENTRY POINT
# =================================================================

def generate_mcqs(text, num_questions=5, difficulty="medium"):
    """
    Generate MCQs from text.

    Uses fine-tuned Flan-T5 on GPU if available (after training),
    otherwise falls back to NLTK NLP pipeline.

    Difficulty: easy / medium / hard
    """
    text = clean_text(text)
    if not text:
        return []

    # Use T5 model if trained and loaded
    if _USE_T5:
        mcqs = _t5_generate_mcqs(text, num_questions, difficulty)
        # If T5 didn't generate enough, supplement with NLTK
        if len(mcqs) < num_questions:
            extra = _nltk_generate(text, num_questions - len(mcqs), difficulty)
            mcqs.extend(extra)
        return mcqs[:num_questions]

    # Fallback: NLTK pipeline
    return _nltk_generate(text, num_questions, difficulty)[:num_questions]
