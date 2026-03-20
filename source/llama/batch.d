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

/// Reset a batch's token count to zero (keeps allocated memory).
void batchClear(ref llama_batch batch) @nogc nothrow
{
    batch.n_tokens = 0;
}

/++
Append one token to a pre-allocated batch (created via `allocBatch`).

Params:
    batch   = target batch; must have been allocated with `allocBatch`
    id      = token id
    pos     = position in the sequence
    seqId   = sequence this token belongs to
    logits  = request logit output for this position
+/
void batchAdd(ref llama_batch batch,
              llama_token     id,
              llama_pos       pos,
              llama_seq_id    seqId,
              bool            logits) @trusted @nogc nothrow
{
    int n = batch.n_tokens;
    batch.token   [n]    = id;
    batch.pos     [n]    = pos;
    batch.n_seq_id[n]    = 1;
    batch.seq_id  [n][0] = seqId;
    batch.logits  [n]    = logits;
    batch.n_tokens       = n + 1;
}
