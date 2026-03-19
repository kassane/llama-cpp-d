module llama.sampling;

import llama.llama;
import llama.ctx : LlamaContext;

/// A sampler chain you configure then use to pick the next token.
struct SamplerChain
{
    private llama_sampler* _smpl;

    @disable this();
    @disable this(this);

    private this(llama_sampler* s) @nogc nothrow { _smpl = s; }

    ~this() @nogc nothrow
    {
        if (_smpl) { llama_sampler_free(_smpl); _smpl = null; }
    }

    /// Create a new sampler chain.
    static SamplerChain create(bool noPerf = false) @nogc nothrow
    {
        auto p = llama_sampler_chain_default_params();
        p.no_perf = noPerf;
        return SamplerChain(llama_sampler_chain_init(p));
    }

    /// Adds greedy (argmax) sampling.
    ref SamplerChain greedy() @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_greedy());
        return this;
    }

    /// Adds temperature scaling.
    ref SamplerChain temp(float t) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_temp(t));
        return this;
    }

    /// Adds top-K filtering.
    ref SamplerChain topK(int k) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_top_k(k));
        return this;
    }

    /// Adds top-P (nucleus) sampling.
    ref SamplerChain topP(float p, size_t minKeep = 1) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_top_p(p, minKeep));
        return this;
    }

    /// Adds min-P sampling.
    ref SamplerChain minP(float p, size_t minKeep = 1) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_min_p(p, minKeep));
        return this;
    }

    /// Adds stochastic (dist) sampling with an optional seed.
    ref SamplerChain dist(uint seed = LLAMA_DEFAULT_SEED) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_dist(seed));
        return this;
    }

    /// Sample the next token. `batchIdx = -1` uses the last output position.
    llama_token sample(llama_context* ctx, int batchIdx = -1) @nogc nothrow
    {
        return llama_sampler_sample(_smpl, ctx, batchIdx);
    }

    /// Sample the next token from a `LlamaContext`.
    llama_token sample(ref LlamaContext ctx, int batchIdx = -1) @nogc nothrow
    {
        return llama_sampler_sample(_smpl, ctx.ptr, batchIdx);
    }

    /// Feed the accepted token back (needed for repetition penalties and similar).
    void accept(llama_token token) @nogc nothrow { llama_sampler_accept(_smpl, token); }

    /// Raw C pointer.
    @property llama_sampler* ptr() @trusted @nogc nothrow { return _smpl; }

    void printPerf() @nogc nothrow { llama_perf_sampler_print(_smpl); }
}
