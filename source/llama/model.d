module llama.model;

import llama.llama;

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
    private llama_model* _model;

    @disable this();
    @disable this(this);

    private this(llama_model* m) @nogc nothrow { _model = m; }

    ~this() @nogc nothrow
    {
        if (_model) { llama_model_free(_model); _model = null; }
    }

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

    @property int nEmbd()  @nogc nothrow { return llama_model_n_embd(_model); }  /// Embedding size.
    @property int nLayer() @nogc nothrow { return llama_model_n_layer(_model); } /// Number of layers.
    @property int nHead()  @nogc nothrow { return llama_model_n_head(_model); }  /// Attention head count.

    @property bool hasEncoder() @nogc nothrow { return llama_model_has_encoder(_model); } /// True for encoder-decoder models (e.g. T5).
    @property bool hasDecoder() @nogc nothrow { return llama_model_has_decoder(_model); }

    /// Start token for the decoder; falls back to BOS for encoder-decoder models.
    @property llama_token decoderStartToken() @nogc nothrow
    {
        llama_token t = llama_model_decoder_start_token(_model);
        if (t == LLAMA_TOKEN_NULL)
            t = llama_vocab_bos(cast(llama_vocab*) llama_model_get_vocab(_model));
        return t;
    }
}
