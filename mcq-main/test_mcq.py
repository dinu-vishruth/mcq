"""Quick test: verify MCQ generation works across all difficulty levels."""
from models.mcq_generator import generate_mcqs

sample_text = """
Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Deep learning is a specialized form of machine learning that uses
neural networks with many layers. Supervised learning requires labeled training data,
while unsupervised learning works with unlabeled data. Python is the most popular
programming language for machine learning. TensorFlow and PyTorch are two widely used
deep learning frameworks developed by Google and Facebook respectively. Natural language
processing is a field of AI that deals with the interaction between computers and human language.
Convolutional neural networks are primarily used for image recognition tasks.
Recurrent neural networks are designed for sequential data like text and speech.
Transfer learning allows models trained on one task to be adapted for another task.
Gradient descent is the optimization algorithm used to train most neural networks.
"""

for difficulty in ["easy", "medium", "hard"]:
    print(f"\n{'='*60}")
    print(f"  DIFFICULTY: {difficulty.upper()}")
    print(f"{'='*60}")
    mcqs = generate_mcqs(sample_text, num_questions=3, difficulty=difficulty)
    for i, q in enumerate(mcqs):
        print(f"\nQ{i+1}: {q['question'][:120]}")
        for o in q["options"]:
            marker = " <-- CORRECT" if o["text"] == q["answer_text"] else ""
            print(f"  {o['label']}) {o['text']}{marker}")

    print(f"\n  Generated: {len(mcqs)} questions")

print("\n\nSUCCESS! All difficulty levels working.")
