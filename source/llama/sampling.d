module llama.sampling;

import llama.llama;
import llama.ctx : LlamaContext;

/// Builder and owner of a `llama_sampler` chain.
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

    /// New sampler chain with default params.
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

    /// Samples the next token from a raw context pointer; `batchIdx = -1` uses the last output position.
    llama_token sample(llama_context* ctx, int batchIdx = -1) @nogc nothrow
    {
        return llama_sampler_sample(_smpl, ctx, batchIdx);
    }

    /// Samples the next token from a LlamaContext wrapper.
    llama_token sample(ref LlamaContext ctx, int batchIdx = -1) @nogc nothrow
    {
        return llama_sampler_sample(_smpl, ctx.ptr, batchIdx);
    }

    /// Notifies the sampler of an accepted token (needed for repetition penalties etc.).
    void accept(llama_token token) @nogc nothrow { llama_sampler_accept(_smpl, token); }

    /// Raw C pointer.
    @property llama_sampler* ptr() @trusted @nogc nothrow { return _smpl; }

    void printPerf() @nogc nothrow { llama_perf_sampler_print(_smpl); }
}
