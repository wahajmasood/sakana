import json

import openai

from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
model = "mistral-nemo"

# Load paper from pdf file (raw text)
paper_txt = load_paper("review_test.pdf")
# Get the review dict of the review
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.1,
)
with open("review.txt", "w") as f:
    f.write(json.dumps(review, indent=4))
