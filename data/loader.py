from sklearn.datasets import fetch_20newsgroups
import re


def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - Remove email headers
    - Remove email addresses
    - Remove quotes
    - Normalize whitespace

    This improves embedding quality by removing metadata noise.
    """

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'>.*\n', '', text)
    text = re.sub(r'\n+', ' ', text)

    return text.strip()


def load_dataset():

    dataset = fetch_20newsgroups(subset="all")

    documents = [clean_text(doc) for doc in dataset.data]

    return documents, dataset.target_names