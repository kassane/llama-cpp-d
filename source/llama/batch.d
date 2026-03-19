module llama.batch;

import llama.llama;

/// A `llama_batch` that frees itself when it goes out of scope.
struct OwnedBatch
{
    private llama_batch _batch;
    private bool _owned;

    @disable this();
    @disable this(this);

    private this(llama_batch b, bool owned) @nogc nothrow
    {
        _batch = b;
        _owned = owned;
    }

    ~this() @nogc nothrow
    {
        if (_owned) { llama_batch_free(_batch); _owned = false; }
    }

    /// Underlying batch.
    ref llama_batch get() @trusted @nogc nothrow { return _batch; }
}

/++ Allocates a batch for up to `nTokensMax` tokens. Pass `embd > 0` for embedding batches. +/
OwnedBatch allocBatch(int nTokensMax, int embd = 0) @nogc nothrow
{
    return OwnedBatch(llama_batch_init(nTokensMax, embd, 1), true);
}

/// Wraps a token slice into a batch. The slice must outlive the returned batch.
llama_batch batchGetOne(scope const(llama_token)[] tokens) @trusted @nogc nothrow
{
    return llama_batch_get_one(cast(llama_token*) tokens.ptr, cast(int) tokens.length);
}

/// Wraps a raw token pointer into a batch; for C interop.
llama_batch batchGetOne(llama_token* tokens, int nTokens) @nogc nothrow
{
    return llama_batch_get_one(tokens, nTokens);
}
