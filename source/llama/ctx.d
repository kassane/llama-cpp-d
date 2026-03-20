module llama.ctx;

import llama.llama;
import llama.model : LlamaModel;
import llama.owned;

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
    mixin Owned!(llama_context, llama_free);

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

    /// Decodes a token batch; returns 0 on success.
    int decode(llama_batch batch) @nogc nothrow { return llama_decode(_ptr, batch); }

    /// Encodes a batch (encoder-decoder models); returns 0 on success.
    int encode(llama_batch batch) @nogc nothrow { return llama_encode(_ptr, batch); }

    /++ Logits at output position `idx` (-1 = last). Valid until the next decode call. +/
    const(float)[] getLogits(int idx = -1) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ptr);
        auto vocab  = llama_model_get_vocab(model);
        int nVocab  = llama_vocab_n_tokens(cast(llama_vocab*) vocab);
        return llama_get_logits_ith(_ptr, idx)[0 .. nVocab];
    }

    @property uint nCtx() @nogc nothrow { return llama_n_ctx(_ptr); } /// Context window size in tokens.

    /// Active pooling type as an int (compare to `LLAMA_POOLING_TYPE_*` constants).
    @property int poolingType() @nogc nothrow { return cast(int) llama_pooling_type(_ptr); }

    /++ Raw memory handle. Use for sequence management (copy, remove, shift, etc.). +/
    @property llama_memory_t memory() @nogc nothrow { return llama_get_memory(_ptr); }

    /// Clear the KV cache. Pass `data = true` to also zero-fill memory.
    void memoryClear(bool data = false) @nogc nothrow
    {
        llama_memory_clear(llama_get_memory(_ptr), data);
    }

    /++
    All output embeddings packed contiguously.
    Valid after `decode`; shape is `[n_outputs * nEmbd]`.
    Returns `null` when pooling is `LLAMA_POOLING_TYPE_NONE` or for generative models.
    +/
    float[] getEmbeddings() @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ptr);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings(_ptr);
        return p ? p[0 .. nEmbd] : null;
    }

    /++ Embeddings for the `i`th output token (-1 = last). Returns `null` for invalid index. +/
    float[] getEmbeddingsIth(int i) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ptr);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings_ith(_ptr, i);
        return p ? p[0 .. nEmbd] : null;
    }

    /++ Pooled embeddings for a sequence. Returns `null` when pooling is `LLAMA_POOLING_TYPE_NONE`. +/
    float[] getEmbeddingsSeq(llama_seq_id seqId) @trusted @nogc nothrow
    {
        auto model  = llama_get_model(_ptr);
        int  nEmbd  = llama_model_n_embd(model);
        float* p = llama_get_embeddings_seq(_ptr, seqId);
        return p ? p[0 .. nEmbd] : null;
    }

    // ── State (session) save / load ──────────────────────────────────────────

    /// Byte count of the current state. Use this to size a buffer before `stateGetData`.
    size_t stateGetSize() @nogc nothrow { return llama_state_get_size(_ptr); }

    /++ Copy the current state into `dst`. Returns the number of bytes written. +/
    size_t stateGetData(ubyte[] dst) @trusted @nogc nothrow
    {
        return llama_state_get_data(_ptr, dst.ptr, dst.length);
    }

    /++ Restore the state from `src`. Returns the number of bytes consumed. +/
    size_t stateSetData(const(ubyte)[] src) @trusted @nogc nothrow
    {
        return llama_state_set_data(_ptr, src.ptr, src.length);
    }

    /++
    Save the state to a session file, recording `tokens` as the session prompt.
    Returns `true` on success.
    +/
    bool stateSaveFile(string path, const(llama_token)[] tokens) @trusted nothrow
    {
        import std.string : toStringz;
        return llama_state_save_file(_ptr, path.toStringz, tokens.ptr, tokens.length);
    }

    /++
    Load state from a session file. On success `tokensOut` is filled and
    `tokenCount` holds the number of tokens read; returns `true`.
    +/
    bool stateLoadFile(string path, llama_token[] tokensOut, scope size_t* tokenCount) @trusted nothrow
    {
        import std.string : toStringz;
        return llama_state_load_file(_ptr, path.toStringz,
                                     tokensOut.ptr, tokensOut.length, tokenCount);
    }

    // ── Per-sequence KV state ────────────────────────────────────────────────

    /// Byte count required to snapshot sequence `seqId`.
    size_t stateSeqGetSize(llama_seq_id seqId) @nogc nothrow
    {
        return llama_state_seq_get_size(_ptr, seqId);
    }

    /++ Copy sequence `seqId`'s KV cache into `dst`. Returns bytes written. +/
    size_t stateSeqGetData(ubyte[] dst, llama_seq_id seqId) @trusted @nogc nothrow
    {
        return llama_state_seq_get_data(_ptr, dst.ptr, dst.length, seqId);
    }

    /++
    Restore a KV snapshot from `src` into sequence `destSeqId`.
    Returns bytes consumed; 0 means failure.
    +/
    size_t stateSeqSetData(const(ubyte)[] src, llama_seq_id destSeqId) @trusted @nogc nothrow
    {
        return llama_state_seq_set_data(_ptr, src.ptr, src.length, destSeqId);
    }

    void printPerf() @nogc nothrow { llama_perf_context_print(_ptr); }
}
