from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
import torch


def load_model_and_tokenizer():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': "cpu"},
        encode_kwargs={'batch_size': 16}
    )

    return tokenizer, model, embeddings, device
