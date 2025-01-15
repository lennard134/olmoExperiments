from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model_id = "allenai/OLMo-2-1124-7B-Instruct-GGUF"
    filename = "olmo-2-1124-7B-instruct-Q8_0.gguf"

    olmo = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
    message = ["Language modeling is "]
    inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
    response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])