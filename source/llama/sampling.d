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

    /++
    Adds repetition / frequency / presence penalties.
    `penaltyLastN = -1` uses the full context; `penaltyLastN = 0` disables the penalty.
    +/
    ref SamplerChain penalties(int penaltyLastN = 64,
                               float penaltyRepeat  = 1.0f,
                               float penaltyFreq    = 0.0f,
                               float penaltyPresent = 0.0f) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl,
            llama_sampler_init_penalties(penaltyLastN, penaltyRepeat,
                                         penaltyFreq, penaltyPresent));
        return this;
    }

    /// Adds typical-P sampling.
    ref SamplerChain typical(float p, size_t minKeep = 1) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_typical(p, minKeep));
        return this;
    }

    /++ Adds temperature sampling with dynamic range extension (`delta` and `exponent`). +/
    ref SamplerChain tempExt(float t, float delta, float exponent) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_temp_ext(t, delta, exponent));
        return this;
    }

    /++ Adds top-N-sigma sampling (keeps tokens within `n` sigma of the top logit). +/
    ref SamplerChain topNSigma(float n) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_top_n_sigma(n));
        return this;
    }

    /++ Adds XTC (exclude top choices) sampling. +/
    ref SamplerChain xtc(float p, float t, size_t minKeep = 1,
                         uint seed = LLAMA_DEFAULT_SEED) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_xtc(p, t, minKeep, seed));
        return this;
    }

    /++ Adds Mirostat v2 sampling (adaptive entropy targeting). +/
    ref SamplerChain mirostatV2(float tau = 5.0f, float eta = 0.1f,
                                uint seed = LLAMA_DEFAULT_SEED) @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl, llama_sampler_init_mirostat_v2(seed, tau, eta));
        return this;
    }

    /++
    Adds grammar-constrained sampling.
    `grammarStr` is a GBNF grammar; `grammarRoot` is the root rule name (usually `"root"`).
    +/
    ref SamplerChain grammar(const(llama_vocab)* vocab,
                             string grammarStr, string grammarRoot = "root") @trusted return
    {
        import std.string : toStringz;
        llama_sampler_chain_add(_smpl,
            llama_sampler_init_grammar(cast(llama_vocab*) vocab,
                                       grammarStr.toStringz, grammarRoot.toStringz));
        return this;
    }

    /++
    Adds DRY (Don't Repeat Yourself) sampling.
    `seqBreakers` lists token strings that reset the repetition check (e.g. `["\n"]`).
    Pass `seqBreakers = []` to use no breakers.
    +/
    ref SamplerChain dry(const(llama_vocab)* vocab,
                         int nCtxTrain,
                         float multiplier    = 0.0f,
                         float base          = 1.75f,
                         int allowedLength   = 2,
                         int penaltyLastN    = -1,
                         string[] seqBreakers = []) @trusted return
    {
        import std.string : toStringz;
        auto cptrs = new const(char)*[](seqBreakers.length);
        foreach (i, s; seqBreakers)
            cptrs[i] = s.toStringz;
        llama_sampler_chain_add(_smpl,
            llama_sampler_init_dry(cast(llama_vocab*) vocab, nCtxTrain, multiplier, base,
                                   allowedLength, penaltyLastN,
                                   cptrs.ptr, cptrs.length));
        return this;
    }

    /++
    Adds per-token logit bias adjustments.
    Each entry in `biases` is a `{token, bias}` pair; `bias > 0` increases probability, `bias < 0` decreases it.
    +/
    ref SamplerChain logitBias(int nVocab, scope const(llama_logit_bias)[] biases) @trusted @nogc nothrow return
    {
        llama_sampler_chain_add(_smpl,
            llama_sampler_init_logit_bias(nVocab, cast(int) biases.length,
                                          cast(llama_logit_bias*) biases.ptr));
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
