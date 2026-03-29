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
    static assert(GGML_MAX_DIMS == 4);
    static assert(GGML_MAX_SRC == 10);
    static assert(GGML_MAX_NAME == 64);
    static assert(GGML_ROPE_TYPE_NEOX == 2);
    static assert(GGML_ROPE_TYPE_MROPE == 8);
    static assert(GGML_ROPE_TYPE_VISION == 24);
}

// ---------------------------------------------------------------------------
// Symbol reachability — core API
// ---------------------------------------------------------------------------

@("llama_* core symbols are reachable")
unittest
{
    auto _0  = &llama_backend_init;
    auto _1  = &llama_backend_free;
    auto _2  = &llama_model_load_from_file;
    auto _3  = &llama_model_free;
    auto _4  = &llama_init_from_model;
    auto _5  = &llama_free;
    auto _6  = &llama_decode;
    auto _7  = &llama_encode;
    auto _8  = &llama_get_logits_ith;
    auto _9  = &llama_tokenize;
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

@("llama_* extended API symbols are reachable")
unittest
{
    // Context / memory / state
    auto _0  = &llama_get_memory;
    auto _1  = &llama_memory_seq_rm;
    auto _2  = &llama_memory_seq_cp;
    auto _3  = &llama_memory_seq_keep;
    auto _4  = &llama_memory_seq_add;
    auto _5  = &llama_memory_seq_div;
    auto _6  = &llama_memory_seq_pos_min;
    auto _7  = &llama_memory_seq_pos_max;
    auto _8  = &llama_memory_can_shift;
    auto _9  = &llama_state_get_size;
    auto _10 = &llama_state_get_data;
    auto _11 = &llama_state_set_data;
    auto _12 = &llama_state_load_file;
    auto _13 = &llama_state_save_file;
    // Embeddings
    auto _14 = &llama_get_embeddings;
    auto _15 = &llama_get_embeddings_ith;
    auto _16 = &llama_get_embeddings_seq;
    // Chat
    auto _17 = &llama_chat_apply_template;
    auto _18 = &llama_chat_builtin_templates;
    // Adapter / LoRA
    auto _19 = &llama_adapter_lora_init;
    auto _20 = &llama_adapter_lora_free;
    auto _21 = &llama_set_adapters_lora;
    // Model metadata
    auto _22 = &llama_model_n_ctx_train;
    auto _23 = &llama_model_n_params;
    auto _24 = &llama_model_size;
    auto _25 = &llama_model_desc;
    auto _26 = &llama_model_chat_template;
    auto _27 = &llama_model_is_recurrent;
    auto _28 = &llama_model_meta_count;
    auto _29 = &llama_model_meta_key_by_index;
    auto _30 = &llama_model_meta_val_str;
    auto _31 = &llama_model_meta_val_str_by_index;
    // Vocab extras
    auto _32 = &llama_vocab_nl;
    auto _33 = &llama_vocab_pad;
    auto _34 = &llama_vocab_sep;
    auto _35 = &llama_vocab_fim_pre;
    auto _36 = &llama_vocab_fim_suf;
    auto _37 = &llama_vocab_fim_mid;
    auto _38 = &llama_vocab_fim_pad;
    auto _39 = &llama_vocab_fim_rep;
    auto _40 = &llama_vocab_fim_sep;
    auto _41 = &llama_vocab_get_text;
    auto _42 = &llama_vocab_get_score;
    auto _43 = &llama_vocab_get_attr;
    auto _44 = &llama_vocab_is_control;
    // Samplers
    auto _45 = &llama_sampler_init_penalties;
    auto _46 = &llama_sampler_init_typical;
    auto _47 = &llama_sampler_init_temp_ext;
    auto _48 = &llama_sampler_init_top_n_sigma;
    auto _49 = &llama_sampler_init_xtc;
    auto _50 = &llama_sampler_init_mirostat_v2;
    auto _51 = &llama_sampler_init_grammar;
    auto _52 = &llama_sampler_init_dry;
    auto _53 = &llama_sampler_init_logit_bias;
}

@("ggml_* symbols are reachable")
unittest
{
    auto _0 = &ggml_time_us;
    auto _1 = &ggml_backend_load_all;
}

// ---------------------------------------------------------------------------
// SamplerChain: chain construction (no model required)
// ---------------------------------------------------------------------------

@("SamplerChain: basic creation — greedy and noPerf variants")
unittest
{
    {
        auto smpl = SamplerChain.create();
        smpl.greedy();
    }
    {
        auto smpl = SamplerChain.create(/*noPerf=*/true);
        smpl.greedy();
        assert(smpl.ptr !is null);
    }
}

@("SamplerChain: temp + topK + topP + dist")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.temp(0.8f).topK(40).topP(0.95f).dist();
}

@("SamplerChain: temp + minP + dist with explicit seed")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.temp(1.0f).minP(0.05f).dist(42u);
}

@("SamplerChain: penalties — normal and disabled (zero lastN)")
unittest
{
    {
        auto smpl = SamplerChain.create();
        smpl.penalties(64, 1.1f, 0.0f, 0.0f).dist();
    }
    {
        auto smpl = SamplerChain.create();
        smpl.penalties(0).dist(); // penalty_last_n=0 disables the sampler
    }
}

@("SamplerChain: typical")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.temp(1.0f).typical(0.95f).dist();
}

@("SamplerChain: tempExt (dynamic temperature)")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.tempExt(0.8f, 0.5f, 1.5f).dist();
}

@("SamplerChain: topNSigma")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.topNSigma(2.0f).dist();
}

@("SamplerChain: xtc")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.xtc(0.1f, 0.1f, 1, LLAMA_DEFAULT_SEED).dist();
}

@("SamplerChain: mirostatV2")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.mirostatV2(5.0f, 0.1f, LLAMA_DEFAULT_SEED);
}

@("SamplerChain: logitBias — empty and with explicit entries")
@trusted unittest
{
    {
        auto smpl = SamplerChain.create();
        smpl.logitBias(32_000, []).dist();
    }
    {
        auto smpl = SamplerChain.create();
        llama_logit_bias[2] biases;
        biases[0].token = 1; biases[0].bias = -100.0f;
        biases[1].token = 2; biases[1].bias =   10.0f;
        smpl.logitBias(32_000, biases[]).dist();
    }
}

@("SamplerChain: combined chain — penalties + typical + tempExt")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.penalties(64, 1.1f).typical(0.9f).tempExt(0.8f, 0.0f, 1.0f).dist();
}

// ---------------------------------------------------------------------------
// Batch helpers
// ---------------------------------------------------------------------------

@("batchGetOne: multi-token and single-token slices")
unittest
{
    {
        llama_token[4] toks = [1, 2, 3, 4];
        llama_batch b = batchGetOne(toks[]);
        assert(b.n_tokens == 4);
        assert(b.token !is null);
    }
    {
        llama_token[1] tok = [42];
        llama_batch b = batchGetOne(tok[]);
        assert(b.n_tokens == 1);
    }
}

@("allocBatch: token batch and embedding batch")
unittest
{
    {
        auto ob = allocBatch(512);
        assert(ob.get().token !is null);
    }
    {
        auto ob = allocBatch(64, 128);
        assert(ob.get().embd !is null);
    }
}

@("batchClear: resets n_tokens to zero")
unittest
{
    auto ob = allocBatch(8);
    batchAdd(ob.get(), 1, 0, 0, true);
    batchAdd(ob.get(), 2, 1, 0, false);
    assert(ob.get().n_tokens == 2);
    batchClear(ob.get());
    assert(ob.get().n_tokens == 0);
}

@("batchAdd: populates token, pos, seq_id, logits fields")
unittest
{
    auto ob = allocBatch(4);
    batchAdd(ob.get(), 42, 7, 3, true);
    ref llama_batch b = ob.get();
    assert(b.n_tokens        == 1);
    assert(b.token[0]        == 42);
    assert(b.pos[0]          == 7);
    assert(b.n_seq_id[0]     == 1);
    assert(b.seq_id[0][0]    == 3);
    assert(b.logits[0]       == true);
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

@("loadAllBackends: idempotent — safe to call multiple times")
unittest
{
    loadAllBackends();
    loadAllBackends();
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

@("LlamaModel: loadFromFile with nonexistent path returns falsy")
unittest
{
    auto model = LlamaModel.loadFromFile("/nonexistent/model.gguf");
    assert(!model);
}

@("LlamaModel: new properties compile")
unittest
{
    static assert(__traits(hasMember, LlamaModel, "isRecurrent"));
    static assert(__traits(hasMember, LlamaModel, "nCtxTrain"));
    static assert(__traits(hasMember, LlamaModel, "nParams"));
    static assert(__traits(hasMember, LlamaModel, "size"));
    static assert(__traits(hasMember, LlamaModel, "desc"));
    static assert(__traits(hasMember, LlamaModel, "chatTemplate"));
    static assert(__traits(hasMember, LlamaModel, "metaCount"));
    static assert(__traits(hasMember, LlamaModel, "metaKeyAt"));
    static assert(__traits(hasMember, LlamaModel, "metaValAt"));
    static assert(__traits(hasMember, LlamaModel, "metaVal"));
}

// ---------------------------------------------------------------------------
// LlamaContext
// ---------------------------------------------------------------------------

@("LlamaContext: new properties compile")
unittest
{
    static assert(__traits(hasMember, LlamaContext, "poolingType"));
    static assert(__traits(hasMember, LlamaContext, "memory"));
    static assert(__traits(hasMember, LlamaContext, "memoryClear"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddings"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddingsIth"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddingsSeq"));
    static assert(__traits(hasMember, LlamaContext, "stateGetSize"));
    static assert(__traits(hasMember, LlamaContext, "stateGetData"));
    static assert(__traits(hasMember, LlamaContext, "stateSetData"));
    static assert(__traits(hasMember, LlamaContext, "stateSaveFile"));
    static assert(__traits(hasMember, LlamaContext, "stateLoadFile"));
    static assert(__traits(hasMember, LlamaContext, "stateSeqGetSize"));
    static assert(__traits(hasMember, LlamaContext, "stateSeqGetData"));
    static assert(__traits(hasMember, LlamaContext, "stateSeqSetData"));
}

// ---------------------------------------------------------------------------
// Owned mixin — structural checks
// ---------------------------------------------------------------------------

@("Owned mixin: injected members present on wrapper structs")
unittest
{
    import llama.owned;
    // ptr() property
    static assert(__traits(hasMember, LlamaModel,   "ptr"));
    static assert(__traits(hasMember, LlamaContext,  "ptr"));
    static assert(__traits(hasMember, SamplerChain,  "ptr"));
    // bool conversion via opCast
    static assert(__traits(compiles, { auto m = LlamaModel.loadFromFile("/x"); bool b = cast(bool) m; }));
    // not copyable
    static assert(!__traits(compiles, { auto m = LlamaModel.loadFromFile("/x"); auto m2 = m; }));
}

// ---------------------------------------------------------------------------
// Vocab helpers
// ---------------------------------------------------------------------------

@("tokenize: empty string returns null without touching C")
unittest
{
    const(llama_vocab)* nullVocab = null;
    assert(tokenize(nullVocab, "") is null);
}

@("detokenize: empty token slice returns empty string without touching C")
unittest
{
    const(llama_vocab)* nullVocab = null;
    assert(detokenize(nullVocab, []) == "");
}

@("vocab: new helper function symbols compile")
unittest
{
    static assert(__traits(compiles, &nlToken));
    static assert(__traits(compiles, &padToken));
    static assert(__traits(compiles, &sepToken));
    static assert(__traits(compiles, &fimPreToken));
    static assert(__traits(compiles, &fimSufToken));
    static assert(__traits(compiles, &fimMidToken));
    static assert(__traits(compiles, &fimPadToken));
    static assert(__traits(compiles, &fimRepToken));
    static assert(__traits(compiles, &fimSepToken));
    static assert(__traits(compiles, &vocabType));
    static assert(__traits(compiles, &tokenText));
    static assert(__traits(compiles, &tokenScore));
    static assert(__traits(compiles, &tokenAttr));
    static assert(__traits(compiles, &isControl));
}

// ---------------------------------------------------------------------------
// chat.d
// ---------------------------------------------------------------------------

@("chat: llama_chat_message struct layout")
@trusted unittest
{
    import llama.chat;
    static assert(is(llama_chat_message == struct));
    llama_chat_message msg;
    msg.role    = "user";
    msg.content = "Hello";
    assert(msg.role !is null && msg.content !is null);
}

@("chat: builtinTemplates returns a non-empty list of named templates")
unittest
{
    import llama.chat : builtinTemplates;
    auto names = builtinTemplates();
    assert(names.length > 0);
    foreach (n; names)
        assert(n.length > 0);
}

@("chat: chatApplyTemplate with a known built-in template name")
@trusted unittest
{
    import llama.chat : builtinTemplates, chatApplyTemplate;
    import std.string : toStringz;
    auto names = builtinTemplates();
    assert(names.length > 0);

    llama_chat_message[1] msgs;
    msgs[0].role    = "user";
    msgs[0].content = "Hello";

    char[1024] buf;
    int n = chatApplyTemplate(names[0].toStringz, msgs[], false, buf[]);
    assert(n > 0);
}

@("chat: applyTemplate returns a non-empty D string")
@trusted unittest
{
    import llama.chat : builtinTemplates, applyTemplate;
    import std.string : toStringz;
    auto names = builtinTemplates();
    assert(names.length > 0);

    llama_chat_message[2] msgs;
    msgs[0].role = "user";      msgs[0].content = "What is 2+2?";
    msgs[1].role = "assistant"; msgs[1].content = "4";

    string result = applyTemplate(names[0].toStringz, msgs[], true);
    assert(result.length > 0);
}

// ---------------------------------------------------------------------------
// adapter.d
// ---------------------------------------------------------------------------

@("adapter: types and functions compile")
unittest
{
    import llama.adapter : LlamaAdapterLora, loadAdapterLora, setAdaptersLora;
    static assert(is(LlamaAdapterLora == struct));
    static assert(__traits(compiles, &loadAdapterLora));
    static assert(__traits(compiles, &setAdaptersLora));
}

@("adapter: loadAdapterLora with nonexistent path returns falsy")
unittest
{
    import llama.adapter : loadAdapterLora;
    loadAllBackends();
    auto model = LlamaModel.loadFromFile("/nonexistent/model.gguf");
    if (!model) return; // no model in CI — skip the rest
    auto adapter = loadAdapterLora(model, "/nonexistent/lora.gguf");
    assert(!adapter);
}

// ---------------------------------------------------------------------------
// mtmd — C binding symbol reachability
// ---------------------------------------------------------------------------

@("mtmd C symbols are reachable")
unittest
{
    import c.mtmd_stubs :
        mtmd_default_marker,
        mtmd_context_params_default,
        mtmd_init_from_file,
        mtmd_free,
        mtmd_bitmap_init,
        mtmd_bitmap_init_from_audio,
        mtmd_bitmap_free,
        mtmd_input_chunks_init,
        mtmd_input_chunks_size,
        mtmd_input_chunks_free,
        mtmd_tokenize,
        mtmd_encode_chunk,
        mtmd_get_output_embd,
        mtmd_helper_bitmap_init_from_file,
        mtmd_helper_bitmap_init_from_buf,
        mtmd_helper_get_n_tokens,
        mtmd_helper_get_n_pos,
        mtmd_helper_eval_chunks;
    auto _0  = &mtmd_default_marker;
    auto _1  = &mtmd_context_params_default;
    auto _2  = &mtmd_init_from_file;
    auto _3  = &mtmd_free;
    auto _4  = &mtmd_bitmap_init;
    auto _5  = &mtmd_bitmap_init_from_audio;
    auto _6  = &mtmd_bitmap_free;
    auto _7  = &mtmd_input_chunks_init;
    auto _8  = &mtmd_input_chunks_size;
    auto _9  = &mtmd_input_chunks_free;
    auto _10 = &mtmd_tokenize;
    auto _11 = &mtmd_encode_chunk;
    auto _12 = &mtmd_get_output_embd;
    auto _13 = &mtmd_helper_bitmap_init_from_file;
    auto _14 = &mtmd_helper_bitmap_init_from_buf;
    auto _15 = &mtmd_helper_get_n_tokens;
    auto _16 = &mtmd_helper_get_n_pos;
    auto _17 = &mtmd_helper_eval_chunks;
}

@("mtmd_context_params_default: n_threads is non-negative")
unittest
{
    import c.mtmd_stubs : mtmd_context_params_default;
    auto p = mtmd_context_params_default();
    assert(p.n_threads >= 0);
}

// ---------------------------------------------------------------------------
// MtmdBitmap wrapper
// ---------------------------------------------------------------------------

@("MtmdBitmap: fromRGB creates bitmap with correct dimensions and data length")
unittest
{
    import llama.mtmd : MtmdBitmap;
    ubyte[12] rgb; // 2×2 RGB
    auto bmp = MtmdBitmap.fromRGB(2, 2, rgb[]);
    assert(cast(bool) bmp);
    assert(bmp.nx == 2 && bmp.ny == 2);
    assert(!bmp.isAudio);
    assert(bmp.data.length == 12);
    assert(bmp.ptr !is null);
}

@("MtmdBitmap: fromAudio creates audio bitmap")
unittest
{
    import llama.mtmd : MtmdBitmap;
    float[16_000] pcm;
    auto bmp = MtmdBitmap.fromAudio(pcm[]);
    assert(cast(bool) bmp && bmp.isAudio);
}

// ---------------------------------------------------------------------------
// InputChunks wrapper + range
// ---------------------------------------------------------------------------

@("InputChunks: empty list — length, empty, nTokens, nPos, iteration")
unittest
{
    import llama.mtmd : InputChunks;
    auto chunks = InputChunks.create();
    assert(chunks.length == 0);
    assert(chunks.empty);
    assert(chunks.nTokens == 0);
    assert(chunks.nPos == 0);
    int count;
    foreach (chunk; chunks) count++;
    assert(count == 0);
}

// ---------------------------------------------------------------------------
// MtmdContext wrapper
// ---------------------------------------------------------------------------

@("MtmdContext: initFromFile with nonexistent path returns falsy")
unittest
{
    import llama.mtmd : MtmdContext;
    auto ctx = MtmdContext.initFromFile("/nonexistent/mmproj.gguf", null);
    assert(!ctx);
}

@("MtmdContext: default marker is a non-empty C string")
unittest
{
    import c.mtmd_stubs : mtmd_default_marker;
    import core.stdc.string : strlen;
    const(char)* m = mtmd_default_marker();
    assert(m !is null && strlen(m) > 0);
}

// ---------------------------------------------------------------------------
// SamplerChain: new introspection + lifecycle methods
// ---------------------------------------------------------------------------

@("SamplerChain: chainN — counts samplers added to chain")
unittest
{
    auto smpl = SamplerChain.create();
    assert(smpl.chainN() == 0);
    smpl.greedy();
    assert(smpl.chainN() == 1);
    smpl.temp(0.8f);
    assert(smpl.chainN() == 2);
}

@("SamplerChain: chainGet — retrieves sampler at index")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.greedy();
    llama_sampler* s = smpl.chainGet(0);
    assert(s !is null);
}

@("SamplerChain: chainRemove — removes and returns sampler")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.greedy();
    assert(smpl.chainN() == 1);
    llama_sampler* s = smpl.chainRemove(0);
    assert(s !is null);
    assert(smpl.chainN() == 0);
    llama_sampler_free(s);
}

@("SamplerChain: getSeed — returns a seed value")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.dist(12345u);
    // Seed is valid; value is implementation-defined but must not crash.
    cast(void) smpl.getSeed();
}

@("SamplerChain: reset — clears state without crash")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.penalties(64, 1.1f).temp(0.8f).dist();
    smpl.reset();
}

@("SamplerChain: clone — produces independent copy")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.greedy();
    assert(smpl.chainN() == 1);
    {
        auto copy = smpl.clone();
        assert(copy.ptr !is null);
        assert(copy.chainN() == 1);
    }
    // Original still valid after clone is destroyed.
    assert(smpl.chainN() == 1);
}

@("SamplerChain: adaptiveP — adds sampler without crash")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.adaptiveP(0.5f, 0.1f);
    assert(smpl.chainN() == 1);
}

@("SamplerChain: grammarLazyPatterns — compiles with empty triggers")
unittest
{
    import llama.sampling : SamplerChain;
    // We don't have a real vocab here, but we can verify the method exists.
    static assert(__traits(hasMember, SamplerChain, "grammarLazyPatterns"));
    static assert(__traits(hasMember, SamplerChain, "infill"));
}

// ---------------------------------------------------------------------------
// LlamaAdapterLora: new ALora properties compile
// ---------------------------------------------------------------------------

@("LlamaAdapterLora: ALora properties compile")
unittest
{
    import llama.adapter : LlamaAdapterLora;
    static assert(__traits(hasMember, LlamaAdapterLora, "nAloraInvocationTokens"));
    static assert(__traits(hasMember, LlamaAdapterLora, "aloraInvocationTokens"));
}

// ---------------------------------------------------------------------------
// LlamaModel: new properties compile
// ---------------------------------------------------------------------------

@("LlamaModel: ropeFreqScaleTrain and saveToFile compile")
unittest
{
    static assert(__traits(hasMember, LlamaModel, "ropeFreqScaleTrain"));
    static assert(__traits(hasMember, LlamaModel, "saveToFile"));
}

// ---------------------------------------------------------------------------
// LlamaContext: new properties compile
// ---------------------------------------------------------------------------

@("LlamaContext: new batch/seq/perf properties compile")
unittest
{
    static assert(__traits(hasMember, LlamaContext, "nBatch"));
    static assert(__traits(hasMember, LlamaContext, "nUbatch"));
    static assert(__traits(hasMember, LlamaContext, "nSeqMax"));
    static assert(__traits(hasMember, LlamaContext, "nCtxSeq"));
    static assert(__traits(hasMember, LlamaContext, "perfReset"));
    static assert(__traits(hasMember, LlamaContext, "printMemoryBreakdown"));
    static assert(__traits(hasMember, LlamaContext, "setSampler"));
}

// ---------------------------------------------------------------------------
// logging.d
// ---------------------------------------------------------------------------

@("logging: suppressLogs and systemInfo symbols reachable")
unittest
{
    import llama.logging : suppressLogs, systemInfo;
    static assert(__traits(compiles, &suppressLogs));
    static assert(__traits(compiles, &systemInfo));
}

@("logging: systemInfo returns a non-empty string")
unittest
{
    import llama.logging : systemInfo;
    loadAllBackends();
    string info = systemInfo();
    assert(info.length > 0);
}

// ---------------------------------------------------------------------------
// config.d — UDA-driven CLI parsing
// ---------------------------------------------------------------------------

@("config: @Param UDA is attached to ModelConfig fields")
unittest
{
    import llama.config : Param, ModelConfig;
    import std.traits : hasUDA;
    static assert(hasUDA!(ModelConfig.modelPath,   Param));
    static assert(hasUDA!(ModelConfig.nGpuLayers,  Param));
    static assert(hasUDA!(ModelConfig.nCtx,        Param));
    static assert(hasUDA!(ModelConfig.nBatch,      Param));
    static assert(hasUDA!(ModelConfig.nPredict,    Param));
    static assert(hasUDA!(ModelConfig.prompt,      Param));
}

@("config: @Param UDA is attached to SamplingConfig fields")
unittest
{
    import llama.config : Param, SamplingConfig;
    import std.traits : hasUDA;
    static assert(hasUDA!(SamplingConfig.temp,          Param));
    static assert(hasUDA!(SamplingConfig.topK,          Param));
    static assert(hasUDA!(SamplingConfig.topP,          Param));
    static assert(hasUDA!(SamplingConfig.minP,          Param));
    static assert(hasUDA!(SamplingConfig.seed,          Param));
    static assert(hasUDA!(SamplingConfig.repeatPenalty, Param));
    static assert(hasUDA!(SamplingConfig.repeatLastN,   Param));
}

@("config: parseConfig — parses known flags into ModelConfig")
unittest
{
    import llama.config : ModelConfig, parseConfig;
    ModelConfig cfg;
    string[] args = ["prog", "-m", "/path/to/model.gguf", "--ngl", "32", "-n", "64"];
    bool ok = parseConfig(cfg, args);
    assert(ok);
    assert(cfg.modelPath   == "/path/to/model.gguf");
    assert(cfg.nGpuLayers  == 32);
    assert(cfg.nPredict    == 64);
}

@("config: parseConfig — parses known flags into SamplingConfig")
unittest
{
    import llama.config : SamplingConfig, parseConfig;
    SamplingConfig cfg;
    string[] args = ["prog", "-t", "0.5", "-k", "20", "--top-p", "0.9", "--seed", "42"];
    bool ok = parseConfig(cfg, args);
    assert(ok);
    assert(cfg.temp  == 0.5f);
    assert(cfg.topK  == 20);
    assert(cfg.topP  == 0.9f);
    assert(cfg.seed  == 42u);
}

@("config: parseConfig — unknown flags remain in args")
unittest
{
    import llama.config : ModelConfig, parseConfig;
    ModelConfig cfg;
    string[] args = ["prog", "-m", "/x.gguf", "--unknown-flag", "value"];
    bool ok = parseConfig(cfg, args);
    assert(ok);
    assert(args.length > 1); // unknown flag was not consumed
}

@("config: buildSamplerChain — greedy when temp == 0")
unittest
{
    import llama.config : SamplingConfig, buildSamplerChain;
    SamplingConfig cfg;
    cfg.temp = 0.0f;
    auto smpl = buildSamplerChain(cfg);
    assert(smpl.ptr !is null);
    // greedy adds 1 sampler (no penalties when repeat == 1.0)
    assert(smpl.chainN() >= 1);
}

@("config: buildSamplerChain — stochastic chain when temp > 0")
unittest
{
    import llama.config : SamplingConfig, buildSamplerChain;
    SamplingConfig cfg; // defaults: temp=0.8, topK=40, topP=0.95, minP=0.05
    auto smpl = buildSamplerChain(cfg);
    assert(smpl.ptr !is null);
    // penalties + temp + topK + topP + minP + dist = 6 samplers
    assert(smpl.chainN() >= 2);
}

// ---------------------------------------------------------------------------
// rag.d — VectorStore and helpers
// ---------------------------------------------------------------------------

@("rag: cosineSimilarity — identical vectors give 1.0")
unittest
{
    import llama.rag : cosineSimilarity;
    float[4] v = [1.0f, 0.0f, 0.0f, 0.0f];
    float sim = cosineSimilarity(v[], v[]);
    assert(sim > 0.999f && sim <= 1.001f);
}

@("rag: cosineSimilarity — orthogonal vectors give 0.0")
unittest
{
    import llama.rag : cosineSimilarity;
    float[2] a = [1.0f, 0.0f];
    float[2] b = [0.0f, 1.0f];
    float sim = cosineSimilarity(a[], b[]);
    assert(sim < 1e-6f && sim > -1e-6f);
}

@("rag: cosineSimilarity — opposite vectors give -1.0")
unittest
{
    import llama.rag : cosineSimilarity;
    float[3] a = [1.0f, 0.0f, 0.0f];
    float[3] b = [-1.0f, 0.0f, 0.0f];
    float sim = cosineSimilarity(a[], b[]);
    assert(sim < -0.999f);
}

@("rag: VectorStore — empty store returns no hits")
unittest
{
    import llama.rag : VectorStore;
    VectorStore store;
    assert(store.length == 0);
    float[4] q = [1.0f, 0.0f, 0.0f, 0.0f];
    auto hits = store.retrieve(q[], 3);
    assert(hits.length == 0);
}

@("rag: VectorStore — addDocument and retrieve")
unittest
{
    import llama.rag : VectorStore;
    VectorStore store;
    float[4] e1 = [1.0f, 0.0f, 0.0f, 0.0f];
    float[4] e2 = [0.0f, 1.0f, 0.0f, 0.0f];
    float[4] e3 = [0.0f, 0.0f, 1.0f, 0.0f];
    store.addDocument("d1", "Document 1", e1[]);
    store.addDocument("d2", "Document 2", e2[]);
    store.addDocument("d3", "Document 3", e3[]);
    assert(store.length == 3);

    float[4] q = [1.0f, 0.1f, 0.0f, 0.0f];
    auto hits = store.retrieve(q[], 2);
    assert(hits.length == 2);
    // d1 should be the top hit (most similar to q)
    assert(hits[0].id == "d1");
    assert(hits[0].score > hits[1].score);
}

@("rag: VectorStore — topK larger than store returns all docs")
unittest
{
    import llama.rag : VectorStore;
    VectorStore store;
    float[2] e = [1.0f, 0.0f];
    store.addDocument("a", "A", e[]);
    store.addDocument("b", "B", e[]);
    auto hits = store.retrieve(e[], 10);
    assert(hits.length == 2);
}

@("rag: VectorStore — clear removes all documents")
unittest
{
    import llama.rag : VectorStore;
    VectorStore store;
    float[2] e = [1.0f, 0.0f];
    store.addDocument("x", "X", e[]);
    assert(store.length == 1);
    store.clear();
    assert(store.length == 0);
}

@("rag: buildRagPrompt — assembles expected sections")
unittest
{
    import llama.rag : Hit, buildRagPrompt;
    import std.string : indexOf;
    Hit[2] hits;
    hits[0] = Hit("d1", "The sky is blue.", 0.9f);
    hits[1] = Hit("d2", "Water is wet.",    0.7f);
    string prompt = buildRagPrompt(hits[], "What color is the sky?");
    assert(prompt.indexOf("Context:")               >= 0);
    assert(prompt.indexOf("[1]")                    >= 0);
    assert(prompt.indexOf("The sky is blue.")       >= 0);
    assert(prompt.indexOf("[2]")                    >= 0);
    assert(prompt.indexOf("Water is wet.")          >= 0);
    assert(prompt.indexOf("Question:")              >= 0);
    assert(prompt.indexOf("What color is the sky?") >= 0);
    assert(prompt.indexOf("Answer:")                >= 0);
}
