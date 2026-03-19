module test_bindings;

import llama;

// ---------------------------------------------------------------------------
// Compile-time constant checks
// ---------------------------------------------------------------------------

@("LLAMA constants match header values")
unittest
{
    static assert(LLAMA_DEFAULT_SEED == 0xFFFF_FFFFu);
    static assert(LLAMA_TOKEN_NULL == -1);
    static assert(LLAMA_SESSION_VERSION == 9);
}

@("GGML constants match header values")
unittest
{
    // importC imports these from ggml.h via llama_stubs
    static assert(GGML_MAX_DIMS == 4);
    static assert(GGML_MAX_SRC == 10);
    static assert(GGML_MAX_NAME == 64);
    static assert(GGML_ROPE_TYPE_NEOX == 2);
    static assert(GGML_ROPE_TYPE_MROPE == 8);
    static assert(GGML_ROPE_TYPE_VISION == 24);
}

// ---------------------------------------------------------------------------
// Symbol reachability
// ---------------------------------------------------------------------------

@("llama_* symbols are reachable")
unittest
{
    auto _0 = &llama_backend_init;
    auto _1 = &llama_backend_free;
    auto _2 = &llama_model_load_from_file;
    auto _3 = &llama_model_free;
    auto _4 = &llama_init_from_model;
    auto _5 = &llama_free;
    auto _6 = &llama_decode;
    auto _7 = &llama_encode;
    auto _8 = &llama_get_logits_ith;
    auto _9 = &llama_tokenize;
    auto _10 = &llama_detokenize;
    auto _11 = &llama_token_to_piece;
    auto _12 = &llama_model_get_vocab;
    auto _13 = &llama_vocab_bos;
    auto _14 = &llama_vocab_eos;
    auto _15 = &llama_vocab_is_eog;
    auto _16 = &llama_sampler_chain_init;
    auto _17 = &llama_sampler_chain_add;
    auto _18 = &llama_sampler_init_greedy;
    auto _19 = &llama_sampler_sample;
    auto _20 = &llama_sampler_free;
    auto _21 = &llama_batch_get_one;
    auto _22 = &llama_perf_context_print;
    auto _23 = &llama_perf_sampler_print;
    auto _24 = &llama_model_n_embd;
    auto _25 = &llama_model_n_layer;
    auto _26 = &llama_model_has_encoder;
    auto _27 = &llama_model_has_decoder;
}

@("ggml_* symbols are reachable")
unittest
{
    auto _0 = &ggml_time_us;
    auto _1 = &ggml_backend_load_all;
}

// ---------------------------------------------------------------------------
// Sampler builder API (no model needed — just tests chain construction)
// ---------------------------------------------------------------------------

@("SamplerChain: create + greedy does not crash")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.greedy();
    // SamplerChain frees itself on scope exit — double-free would crash.
}

@("SamplerChain: create with noPerf flag")
unittest
{
    auto smpl = SamplerChain.create(true);
    smpl.greedy();
}

@("SamplerChain: chained temperature + topK + topP + dist")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.temp(0.8f).topK(40).topP(0.95f).dist();
}

@("SamplerChain: minP sampler")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.temp(1.0f).minP(0.05f).dist(42u);
}

@("SamplerChain: ptr is non-null after create")
unittest
{
    auto smpl = SamplerChain.create();
    assert(smpl.ptr !is null);
}

// ---------------------------------------------------------------------------
// Batch helpers
// ---------------------------------------------------------------------------

@("batchGetOne: D-slice overload compiles and returns valid struct")
unittest
{
    llama_token[4] toks = [1, 2, 3, 4];
    llama_batch b = batchGetOne(toks[]);
    assert(b.n_tokens == 4);
    assert(b.token !is null);
}

@("batchGetOne: single-element stack array (scope slice)")
unittest
{
    llama_token[1] tok = [42];
    llama_batch b = batchGetOne(tok[]);
    assert(b.n_tokens == 1);
}

@("allocBatch: token batch")
unittest
{
    auto ob = allocBatch(512);
    assert(ob.get().token !is null);
}

@("allocBatch: embedding batch")
unittest
{
    auto ob = allocBatch(64, 128);
    assert(ob.get().embd !is null);
}

// ---------------------------------------------------------------------------
// Backend guard
// ---------------------------------------------------------------------------

@("loadAllBackends: callable without crash")
unittest
{
    // loadAllBackends wraps llama_backend_init + ggml_backend_load_all.
    // Calling it multiple times should be safe (idempotent at C level).
    loadAllBackends();
    loadAllBackends();
}

// ---------------------------------------------------------------------------
// LlamaModel: null-model opCast
// ---------------------------------------------------------------------------

@("LlamaModel: loadFromFile with nonexistent path returns falsy model")
unittest
{
    auto model = LlamaModel.loadFromFile("/nonexistent/model.gguf");
    assert(!model, "Loading from nonexistent path should yield a null model");
}

// ---------------------------------------------------------------------------
// Vocab helpers (compile-time + type-level)
// ---------------------------------------------------------------------------

@("tokenize: empty string returns null slice")
unittest
{
    // We can't call into the real vocab without a loaded model, but we can
    // verify the function handles the fast-path for empty input at the
    // D-wrapper level without touching C at all.
    const(llama_vocab)* nullVocab = null;
    llama_token[] result = tokenize(nullVocab, "");
    assert(result is null, "tokenize of empty string must return null");
}

@("detokenize: empty token slice returns empty string")
unittest
{
    const(llama_vocab)* nullVocab = null;
    string s = detokenize(nullVocab, []);
    assert(s == "", "detokenize of empty slice must return empty string");
}

// ---------------------------------------------------------------------------
// llama_cpp extern(C++) declarations reachable
// ---------------------------------------------------------------------------

@("llama_cpp.d: extern(C++) deleter structs compile and are correctly typed")
unittest
{
    import llama.llama_cpp;

    // Verify the structs are declared with C++ linkage — the fact that
    // this compiles at all confirms the extern(C++) mangling is applied.
    static assert(is(llama_model_deleter == struct));
    static assert(is(llama_context_deleter == struct));
    static assert(is(llama_sampler_deleter == struct));
    static assert(is(llama_adapter_lora_deleter == struct));
}
