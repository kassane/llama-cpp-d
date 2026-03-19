module llama.adapter;

import llama.llama;
import llama.model : LlamaModel;
import llama.ctx  : LlamaContext;

/++
A LoRA adapter loaded from a GGUF file.
Freed automatically on destruction; you may also call `free()` early.
The associated model must remain alive for the adapter's lifetime.
+/
struct LlamaAdapterLora
{
    private llama_adapter_lora* _adapter;

    @disable this();
    @disable this(this);

    private this(llama_adapter_lora* a) @nogc nothrow { _adapter = a; }

    ~this() @nogc nothrow
    {
        if (_adapter) { llama_adapter_lora_free(_adapter); _adapter = null; }
    }

    bool opCast(T : bool)() @nogc nothrow { return _adapter !is null; }

    /// Raw C pointer.
    @property llama_adapter_lora* ptr() @trusted @nogc nothrow { return _adapter; }

    /++ Number of GGUF metadata entries. +/
    @property int metaCount() @nogc nothrow { return llama_adapter_meta_count(_adapter); }

    /// Metadata value by key. Returns `""` on failure.
    string metaVal(string key) @trusted
    {
        import std.string : toStringz;
        char[4096] buf;
        int n = llama_adapter_meta_val_str(_adapter, key.toStringz, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata key name at `index`. Returns `""` on failure.
    string metaKeyAt(int index) @trusted
    {
        char[512] buf;
        int n = llama_adapter_meta_key_by_index(_adapter, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata value at `index`. Returns `""` on failure.
    string metaValAt(int index) @trusted
    {
        char[4096] buf;
        int n = llama_adapter_meta_val_str_by_index(_adapter, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Free the adapter early (safe to call multiple times).
    void free() @nogc nothrow
    {
        if (_adapter) { llama_adapter_lora_free(_adapter); _adapter = null; }
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
