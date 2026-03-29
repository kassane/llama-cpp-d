module llama.adapter;

import llama.llama;
import llama.model : LlamaModel;
import llama.ctx  : LlamaContext;
import llama.owned;

/++
A LoRA adapter loaded from a GGUF file.
Freed automatically on destruction.
The associated model must remain alive for the adapter's lifetime.
+/
struct LlamaAdapterLora
{
    mixin Owned!(llama_adapter_lora, llama_adapter_lora_free);

    /++ Number of GGUF metadata entries. +/
    @property int metaCount() @nogc nothrow { return llama_adapter_meta_count(_ptr); }

    /// Metadata value by key. Returns `""` on failure.
    string metaVal(string key) @trusted
    {
        import std.string : toStringz;
        char[4096] buf;
        int n = llama_adapter_meta_val_str(_ptr, key.toStringz, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata key name at `index`. Returns `""` on failure.
    string metaKeyAt(int index) @trusted
    {
        char[512] buf;
        int n = llama_adapter_meta_key_by_index(_ptr, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata value at `index`. Returns `""` on failure.
    string metaValAt(int index) @trusted
    {
        char[4096] buf;
        int n = llama_adapter_meta_val_str_by_index(_ptr, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    // ── ALora (activation LoRA) ───────────────────────────────────────────────

    /++
    Number of invocation tokens stored in the adapter when it is an ALora adapter.
    Returns 0 for ordinary LoRA adapters.
    +/
    @property ulong nAloraInvocationTokens() @nogc nothrow
    {
        return llama_adapter_get_alora_n_invocation_tokens(_ptr);
    }

    /++
    Invocation token sequence for ALora adapters.
    Returns an empty slice for ordinary LoRA adapters.
    The memory is owned by the adapter; do not free or outlive it.
    +/
    @property const(llama_token)[] aloraInvocationTokens() @trusted @nogc nothrow
    {
        ulong n = nAloraInvocationTokens();
        if (n == 0) return null;
        return llama_adapter_get_alora_invocation_tokens(_ptr)[0 .. n];
    }
}

/// Load a LoRA adapter from a GGUF file. Check `if (adapter)` after loading.
LlamaAdapterLora loadAdapterLora(ref LlamaModel model, string path)
{
    import std.string : toStringz;
    return LlamaAdapterLora(llama_adapter_lora_init(model.ptr, path.toStringz));
}

/++
Apply a set of LoRA adapters to a context.
`adapters` is a slice of raw C pointers (use `adapter.ptr` on each `LlamaAdapterLora`).
`scales` must have the same length as `adapters`.
Pass an empty slice to clear all adapters.
Returns 0 on success.
+/
int setAdaptersLora(ref LlamaContext ctx,
                    llama_adapter_lora*[] adapters,
                    float[] scales) @trusted @nogc nothrow
in (adapters.length == scales.length)
{
    return llama_set_adapters_lora(ctx.ptr, adapters.ptr, adapters.length, scales.ptr);
}
