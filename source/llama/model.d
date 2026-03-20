module llama.model;

import llama.llama;
import llama.owned;

/// Default model params with `nGpuLayers` GPU layers.
llama_model_params modelParams(int nGpuLayers = 0) @nogc nothrow
{
    auto p = llama_model_default_params();
    p.n_gpu_layers = nGpuLayers;
    return p;
}

/// A loaded llama model that frees itself on destruction.
struct LlamaModel
{
    mixin Owned!(llama_model, llama_model_free);

    /// Load from a GGUF file with pre-built params. Check `if (model)` after loading.
    static LlamaModel loadFromFile(const(char)* path, llama_model_params params) @nogc nothrow
    {
        return LlamaModel(llama_model_load_from_file(path, params));
    }

    /// Load from a D string path, with optional GPU layer count.
    static LlamaModel loadFromFile(string path, int nGpuLayers = 0)
    {
        import std.string : toStringz;
        return LlamaModel(llama_model_load_from_file(path.toStringz, modelParams(nGpuLayers)));
    }

    /// Load only the vocabulary (no weights). Useful for tokenization without inference.
    static LlamaModel loadVocabOnly(string path)
    {
        import std.string : toStringz;
        auto p = modelParams(0);
        p.vocab_only = true;
        return LlamaModel(llama_model_load_from_file(path.toStringz, p));
    }

    /// Model vocabulary.
    @property const(llama_vocab)* vocab() @nogc nothrow { return llama_model_get_vocab(_ptr); }

    /// Number of tokens in the vocabulary.
    @property int nVocab() @nogc nothrow
    {
        return llama_vocab_n_tokens(cast(llama_vocab*) llama_model_get_vocab(_ptr));
    }

    @property int nEmbd()  @nogc nothrow { return llama_model_n_embd(_ptr); }  /// Embedding size.
    @property int nLayer() @nogc nothrow { return llama_model_n_layer(_ptr); } /// Number of layers.
    @property int nHead()  @nogc nothrow { return llama_model_n_head(_ptr); }  /// Attention head count.

    @property bool hasEncoder()  @nogc nothrow { return llama_model_has_encoder(_ptr); }  /// True for encoder-decoder models (e.g. T5).
    @property bool hasDecoder()  @nogc nothrow { return llama_model_has_decoder(_ptr); }
    @property bool isRecurrent() @nogc nothrow { return llama_model_is_recurrent(_ptr); } /// True for recurrent models (Mamba, RWKV, etc.).

    /// Start token for the decoder; falls back to BOS for encoder-decoder models.
    @property llama_token decoderStartToken() @nogc nothrow
    {
        llama_token t = llama_model_decoder_start_token(_ptr);
        if (t == LLAMA_TOKEN_NULL)
            t = llama_vocab_bos(cast(llama_vocab*) llama_model_get_vocab(_ptr));
        return t;
    }

    @property int    nCtxTrain() @nogc nothrow { return llama_model_n_ctx_train(_ptr); } /// Training context length.
    @property ulong  nParams()   @nogc nothrow { return llama_model_n_params(_ptr); }    /// Total parameter count.
    @property ulong  size()      @nogc nothrow { return llama_model_size(_ptr); }        /// Model size in bytes.

    /// Short description string (architecture + size).
    @property string desc() @trusted
    {
        char[256] buf;
        int n = llama_model_desc(_ptr, buf.ptr, buf.length);
        return n > 0 ? buf[0 .. n].idup : "";
    }

    /++
    Jinja chat template embedded in the model (or the named variant).
    Returns `null` if none is available.
    Pass `name = null` for the default template.
    +/
    const(char)* chatTemplate(const(char)* name = null) @trusted @nogc nothrow
    {
        return llama_model_chat_template(_ptr, name);
    }

    // ── Metadata access ───────────────────────────────────────────────────────

    /// Number of key/value metadata pairs.
    @property int metaCount() @nogc nothrow { return llama_model_meta_count(_ptr); }

    /// Metadata key name at `index`. Returns `""` on failure.
    string metaKeyAt(int index) @trusted
    {
        char[512] buf;
        int n = llama_model_meta_key_by_index(_ptr, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata value (as string) at `index`. Returns `""` on failure.
    string metaValAt(int index) @trusted
    {
        char[4096] buf;
        int n = llama_model_meta_val_str_by_index(_ptr, index, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }

    /// Metadata value (as string) for the given `key`. Returns `""` on failure.
    string metaVal(string key) @trusted
    {
        import std.string : toStringz;
        char[4096] buf;
        int n = llama_model_meta_val_str(_ptr, key.toStringz, buf.ptr, buf.length);
        return n >= 0 ? buf[0 .. n].idup : "";
    }
}
