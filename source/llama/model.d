module llama.model;

import llama.llama;

/// Returns model params with `nGpuLayers` GPU layers and all other settings at their defaults.
llama_model_params modelParams(int nGpuLayers = 0) @nogc nothrow
{
    auto p = llama_model_default_params();
    p.n_gpu_layers = nGpuLayers;
    return p;
}

/// Owns a `llama_model*`, frees it on destruction.
struct LlamaModel
{
    private llama_model* _model;

    @disable this();
    @disable this(this);

    private this(llama_model* m) @nogc nothrow { _model = m; }

    ~this() @nogc nothrow
    {
        if (_model) { llama_model_free(_model); _model = null; }
    }

    /// Loads a model from a GGUF file (raw C path + pre-built params). Check `if (model)` after.
    static LlamaModel loadFromFile(const(char)* path, llama_model_params params) @nogc nothrow
    {
        return LlamaModel(llama_model_load_from_file(path, params));
    }

    /// Convenience overload: takes a D string and optional GPU layer count.
    static LlamaModel loadFromFile(string path, int nGpuLayers = 0)
    {
        import std.string : toStringz;
        return LlamaModel(llama_model_load_from_file(path.toStringz, modelParams(nGpuLayers)));
    }

    /// Loads only the vocabulary (no weight tensors); useful for tokenization without inference.
    static LlamaModel loadVocabOnly(string path)
    {
        import std.string : toStringz;
        auto p = modelParams(0);
        p.vocab_only = true;
        return LlamaModel(llama_model_load_from_file(path.toStringz, p));
    }

    bool opCast(T : bool)() @nogc nothrow { return _model !is null; }

    /// Raw C pointer.
    @property llama_model* ptr() @trusted @nogc nothrow { return _model; }

    /// Model vocabulary.
    @property const(llama_vocab)* vocab() @nogc nothrow { return llama_model_get_vocab(_model); }

    /// Number of tokens in the vocabulary.
    @property int nVocab() @nogc nothrow
    {
        return llama_vocab_n_tokens(cast(llama_vocab*) llama_model_get_vocab(_model));
    }

    @property int nEmbd()  @nogc nothrow { return llama_model_n_embd(_model); }  /// Embedding dimensions.
    @property int nLayer() @nogc nothrow { return llama_model_n_layer(_model); } /// Layer count.
    @property int nHead()  @nogc nothrow { return llama_model_n_head(_model); }  /// Attention heads.

    @property bool hasEncoder() @nogc nothrow { return llama_model_has_encoder(_model); } /// True for encoder-decoder models (e.g. T5).
    @property bool hasDecoder() @nogc nothrow { return llama_model_has_decoder(_model); }

    /// First token fed to the decoder in encoder-decoder models; falls back to BOS.
    @property llama_token decoderStartToken() @nogc nothrow
    {
        llama_token t = llama_model_decoder_start_token(_model);
        if (t == LLAMA_TOKEN_NULL)
            t = llama_vocab_bos(cast(llama_vocab*) llama_model_get_vocab(_model));
        return t;
    }
}
