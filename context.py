"""Pack the same evidence text for all four model tokenizers."""
from config import PREFIX, SEPARATOR
from common import object_hash


def make_prompt(query, context):
    return f"{PREFIX}\n\nQuestion:{query}\n\nContext:\n\n{context}"


def chat_ids(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if not isinstance(encoded, list) or (encoded and isinstance(encoded[0], list)):
        raise ValueError("Expected a single tokenized chat prompt.")
    return encoded


def tokenizer_signature(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    representation = backend.to_str() if backend is not None else tokenizer.get_vocab()
    return object_hash({
        "class": type(tokenizer).__name__, "representation": representation,
        "chat_template": tokenizer.chat_template,
        "special_tokens": tokenizer.special_tokens_map,
    })


def pack_context(query, retrieved, tokenizers, context_limit=4096, max_new_tokens=64):
    if not tokenizers or context_limit <= max_new_tokens or max_new_tokens <= 0:
        raise ValueError("Invalid tokenizer list or token budget.")
    budget = context_limit - max_new_tokens

    def lengths(text):
        prompt = make_prompt(query, text)
        return {name: len(chat_ids(tokenizer, prompt))
                for name, tokenizer in tokenizers.items()}

    def fits(text):
        return max(lengths(text).values()) <= budget

    if not fits(""):
        raise ValueError("The instruction and question alone exceed the input budget.")
    parts, used, full_count, partial = [], [], 0, False
    for item in retrieved[:10]:
        candidate = SEPARATOR.join(parts + [item["text"]])
        if fits(candidate):
            parts.append(item["text"])
            used.append(item.get("chunk_id"))
            full_count += 1
            continue
        # Keep the complete metadata header if this is a formatted course chunk.
        header, separator, body = item["text"].partition("\n\n")
        if not separator:
            header, body = "", item["text"]
        header = header + "\n\n" if header else ""
        words = body.split()
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            text = SEPARATOR.join(parts + [header + " ".join(words[:middle])])
            if fits(text):
                low = middle
            else:
                high = middle - 1
        # Token counts need not be perfectly monotone around a word boundary.
        while low and not fits(SEPARATOR.join(parts + [header + " ".join(words[:low])])):
            low -= 1
        if low:
            parts.append(header + " ".join(words[:low]))
            used.append(item.get("chunk_id"))
            partial = True
        break
    result = SEPARATOR.join(parts)
    final_lengths = lengths(result)
    if any(length > budget for length in final_lengths.values()):
        raise ValueError("Final chat-template budget validation failed.")
    return {
        "context": result, "prompt": make_prompt(query, result),
        "input_tokens": final_lengths, "full_chunks": full_count,
        "partial_chunk": partial, "used_chunk_ids": used,
        "retrieved_chunks": min(10, len(retrieved)),
        "truncated": full_count < min(10, len(retrieved)),
        "context_sha256": object_hash(result),
    }
