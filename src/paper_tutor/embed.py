"""Embedding and reranking model helpers shared by all scripts."""

from sentence_transformers import CrossEncoder, SentenceTransformer


def active_model(config):
    """Return the short name and the settings of the model chosen in the config."""
    key = config["embeddings"]["active"]
    return key, config["embeddings"]["models"][key]


def load_model(model_cfg):
    model = SentenceTransformer(model_cfg["name"])
    if "max_seq_length" in model_cfg:
        model.max_seq_length = model_cfg["max_seq_length"]
    return model


def encode_query(model, model_cfg, text):
    """Encode a search question, adding the model's query prompt if it has one."""
    if "query_prompt" in model_cfg:
        return model.encode(text, prompt=model_cfg["query_prompt"], normalize_embeddings=True)
    if "query_prompt_name" in model_cfg:
        return model.encode(text, prompt_name=model_cfg["query_prompt_name"], normalize_embeddings=True)
    return model.encode(text, normalize_embeddings=True)


def load_reranker(name):
    """Cross-encoder that scores a (question, paper text) pair directly; slower but more precise."""
    return CrossEncoder(name)
