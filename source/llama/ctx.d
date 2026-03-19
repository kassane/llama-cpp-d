module llama.ctx;

import llama.llama;
import llama.model : LlamaModel;

/++ Context params with the given window and batch size. `nCtx = 0` uses the model's training length. +/
llama_context_params contextParams(uint nCtx = 0, uint nBatch = 512, bool noPerf = false) @nogc nothrow
{
    auto p = llama_context_default_params();
    p.n_ctx   = nCtx;
    p.n_batch = nBatch;
    p.no_perf = noPerf;
    return p;
}

/// A `llama_context` that frees itself on destruction.
struct LlamaContext
{
    private llama_context* _ctx;

    @disable this();
    @disable this(this);

    private this(llama_context* c) @nogc nothrow { _ctx = c; }

    ~this() @nogc nothrow
    {
        if (_ctx) { llama_free(_ctx); _ctx = null; }
    }

    /// Create from explicit params.
    static LlamaContext fromModel(ref LlamaModel model, llama_context_params params) @nogc nothrow
    {
        return LlamaContext(llama_init_from_model(model.ptr, params));
    }

    /// Create from a window size and batch size.
    static LlamaContext fromModel(ref LlamaModel model, uint nCtx, uint nBatch = 512) @nogc nothrow
    {
        return LlamaContext(llama_init_from_model(model.ptr, contextParams(nCtx, nBatch)));
    }

    bool opCast(T : bool)() @nogc nothrow { return _ctx !is null; }

    /// Raw C pointer.
    @property llama_context* ptr() @trusted @nogc nothrow { return _ctx; }

    /// Decodes a token batch; returns 0 on success.
    int decode(llama_batch batch) @nogc nothrow { return llama_decode(_ctx, batch); }

    /// Encodes a batch (encoder-decoder models); returns 0 on success.
    int encode(llama_batch batch) @nogc nothrow { return llama_encode(_ctx, batch); }

    /++ Logits at output position `idx` (-1 = last). Valid until the next decode call. +/
    const(float)[] getLogits(int idx = -1) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ctx);
        auto vocab  = llama_model_get_vocab(model);
        int nVocab  = llama_vocab_n_tokens(cast(llama_vocab*) vocab);
        return llama_get_logits_ith(_ctx, idx)[0 .. nVocab];
    }

    @property uint nCtx() @nogc nothrow { return llama_n_ctx(_ctx); } /// Context window size in tokens.

    /// Active pooling type as an int (compare to `LLAMA_POOLING_TYPE_*` constants).
    @property int poolingType() @nogc nothrow { return cast(int) llama_pooling_type(_ctx); }

    /++ Raw memory handle. Use for sequence management (copy, remove, shift, etc.). +/
    @property llama_memory_t memory() @nogc nothrow { return llama_get_memory(_ctx); }

    /++
    All output embeddings packed contiguously.
    Valid after `decode`; shape is `[n_outputs * nEmbd]`.
    Returns `null` when pooling is `LLAMA_POOLING_TYPE_NONE` or for generative models.
    +/
    float[] getEmbeddings() @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ctx);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings(_ctx);
        return p ? p[0 .. nEmbd] : null;
    }

    /++ Embeddings for the `i`th output token (-1 = last). Returns `null` for invalid index. +/
    float[] getEmbeddingsIth(int i) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ctx);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings_ith(_ctx, i);
        return p ? p[0 .. nEmbd] : null;
    }

    /++ Pooled embeddings for a sequence. Returns `null` when pooling is `LLAMA_POOLING_TYPE_NONE`. +/
    float[] getEmbeddingsSeq(llama_seq_id seqId) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ctx);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings_seq(_ctx, seqId);
        return p ? p[0 .. nEmbd] : null;
    }

    // ── State (session) save / load ──────────────────────────────────────────

    /// Byte count of the current state. Use this to size a buffer before `stateGetData`.
    size_t stateGetSize() @nogc nothrow { return llama_state_get_size(_ctx); }

    /++ Copy the current state into `dst`. Returns the number of bytes written. +/
    size_t stateGetData(ubyte[] dst) @trusted @nogc nothrow
    {
        return llama_state_get_data(_ctx, dst.ptr, dst.length);
    }

    /++ Restore the state from `src`. Returns the number of bytes consumed. +/
    size_t stateSetData(const(ubyte)[] src) @trusted @nogc nothrow
    {
        return llama_state_set_data(_ctx, src.ptr, src.length);
    }

    /++
    Save the state to a session file, recording `tokens` as the session prompt.
    Returns `true` on success.
    +/
    bool stateSaveFile(string path, const(llama_token)[] tokens) @trusted nothrow
    {
        import std.string : toStringz;
        return llama_state_save_file(_ctx, path.toStringz, tokens.ptr, tokens.length);
    }

    /++
    Load state from a session file. On success `tokensOut` is filled and
    `tokenCount` holds the number of tokens read; returns `true`.
    +/
    bool stateLoadFile(string path, llama_token[] tokensOut, size_t* tokenCount) @trusted nothrow
    {
        import std.string : toStringz;
        return llama_state_load_file(_ctx, path.toStringz,
                                     tokensOut.ptr, tokensOut.length, tokenCount);
    }

    void printPerf() @nogc nothrow { llama_perf_context_print(_ctx); }
}
