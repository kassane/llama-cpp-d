module llama.ctx;

import llama.llama;
import llama.model : LlamaModel;

/// Returns context params with the given window size and batch size; everything else at defaults.
/// Pass `nCtx = 0` to use the model's training context length.
llama_context_params contextParams(uint nCtx = 0, uint nBatch = 512, bool noPerf = false) @nogc nothrow
{
    auto p = llama_context_default_params();
    p.n_ctx   = nCtx;
    p.n_batch = nBatch;
    p.no_perf = noPerf;
    return p;
}

/// Owns a `llama_context*`, frees it on destruction.
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

    /// Creates a context from a loaded model with explicit params.
    static LlamaContext fromModel(ref LlamaModel model, llama_context_params params) @nogc nothrow
    {
        return LlamaContext(llama_init_from_model(model.ptr, params));
    }

    /// Convenience overload: takes a window size and batch size directly.
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

    /// Logits for output at `idx` (-1 = last). Slice is valid until the next decode.
    /// The returned slice is read-only; the underlying memory is owned by the context.
    const(float)[] getLogits(int idx = -1) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ctx);
        auto vocab  = llama_model_get_vocab(model);
        int nVocab  = llama_vocab_n_tokens(cast(llama_vocab*) vocab);
        return llama_get_logits_ith(_ctx, idx)[0 .. nVocab];
    }

    @property uint nCtx() @nogc nothrow { return llama_n_ctx(_ctx); } /// Context window size in tokens.

    void printPerf() @nogc nothrow { llama_perf_context_print(_ctx); }
}
