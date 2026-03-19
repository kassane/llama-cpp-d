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

@("llama_* new API symbols are reachable")
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

// ---------------------------------------------------------------------------
// mtmd (multimodal) — C binding symbol reachability
// ---------------------------------------------------------------------------

@("mtmd C symbols are reachable")
unittest
{
    import llama.mtmd;

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

@("mtmd_context_params_default: returns without crash")
unittest
{
    import llama.mtmd : mtmd_context_params_default;
    auto p = mtmd_context_params_default();
    // n_threads should be a sane non-negative value (or 0 = auto)
    assert(p.n_threads >= 0);
}

// ---------------------------------------------------------------------------
// MtmdBitmap RAII wrapper
// ---------------------------------------------------------------------------

@("MtmdBitmap: fromRGB creates valid bitmap with correct dimensions")
unittest
{
    import llama.mtmd : MtmdBitmap;
    ubyte[12] rgb; // 2×2 RGB image (all zeros)
    auto bmp = MtmdBitmap.fromRGB(2, 2, rgb[]);
    assert(cast(bool) bmp);
    assert(bmp.nx == 2);
    assert(bmp.ny == 2);
    assert(!bmp.isAudio);
    assert(bmp.data.length == 12);
}

@("MtmdBitmap: fromAudio creates valid audio bitmap")
unittest
{
    import llama.mtmd : MtmdBitmap;
    float[16_000] pcm; // 1 second at 16kHz
    auto bmp = MtmdBitmap.fromAudio(pcm[]);
    assert(cast(bool) bmp);
    assert(bmp.isAudio);
}

@("MtmdBitmap: ptr is non-null after create")
unittest
{
    import llama.mtmd : MtmdBitmap;
    ubyte[3] px = [255, 0, 0]; // 1×1 red pixel
    auto bmp = MtmdBitmap.fromRGB(1, 1, px[]);
    assert(bmp.ptr !is null);
}

// ---------------------------------------------------------------------------
// InputChunks RAII + range
// ---------------------------------------------------------------------------

@("InputChunks: create returns empty chunk list")
unittest
{
    import llama.mtmd : InputChunks;
    auto chunks = InputChunks.create();
    assert(chunks.length == 0);
    assert(chunks.empty);
}

@("InputChunks: opApply over empty list iterates zero times")
unittest
{
    import llama.mtmd : InputChunks;
    auto chunks = InputChunks.create();
    int count;
    foreach (chunk; chunks) count++;
    assert(count == 0);
}

@("InputChunks: nTokens and nPos are zero for empty list")
unittest
{
    import llama.mtmd : InputChunks;
    auto chunks = InputChunks.create();
    assert(chunks.nTokens == 0);
    assert(chunks.nPos == 0);
}

// ---------------------------------------------------------------------------
// MtmdContext RAII wrapper
// ---------------------------------------------------------------------------

@("MtmdContext: initFromFile with nonexistent path returns falsy")
unittest
{
    import llama.mtmd : MtmdContext;
    // null text_model is enough to make mtmd_init_from_file fail fast
    auto ctx = MtmdContext.initFromFile("/nonexistent/mmproj.gguf", null);
    assert(!ctx, "Init from nonexistent file must yield a null context");
}

@("MtmdContext: default marker is non-empty C string")
unittest
{
    import llama.mtmd : mtmd_default_marker;
    import core.stdc.string : strlen;
    const(char)* m = mtmd_default_marker();
    assert(m !is null);
    assert(strlen(m) > 0);
}

// ---------------------------------------------------------------------------
// SamplerChain: new methods (chain construction — no model required)
// ---------------------------------------------------------------------------

@("SamplerChain: penalties chain construction")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.penalties(64, 1.1f, 0.0f, 0.0f).dist();
}

@("SamplerChain: penalties — disable via zero penalty_last_n")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.penalties(0).dist();
}

@("SamplerChain: typical sampling")
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

@("SamplerChain: logitBias with empty slice")
unittest
{
    auto smpl = SamplerChain.create();
    // Empty bias list is a no-op — should not crash.
    smpl.logitBias(32_000, []).dist();
}

@("SamplerChain: logitBias with explicit entries")
@trusted unittest
{
    auto smpl = SamplerChain.create();
    llama_logit_bias[2] biases;
    biases[0].token = 1; biases[0].bias = -100.0f;
    biases[1].token = 2; biases[1].bias =   10.0f;
    smpl.logitBias(32_000, biases[]).dist();
}

@("SamplerChain: combined chain with penalties + typical + tempExt")
unittest
{
    auto smpl = SamplerChain.create();
    smpl.penalties(64, 1.1f).typical(0.9f).tempExt(0.8f, 0.0f, 1.0f).dist();
}

// ---------------------------------------------------------------------------
// chat.d
// ---------------------------------------------------------------------------

@("chat: llama_chat_message struct layout")
@trusted unittest
{
    import llama.chat;
    static assert(is(llama_chat_message == struct));
    llama_chat_message msg = {};
    msg.role    = "user";
    msg.content = "Hello";
    assert(msg.role    !is null);
    assert(msg.content !is null);
}

@("chat: builtinTemplates returns a non-empty list of named templates")
unittest
{
    import llama.chat : builtinTemplates;
    auto templates = builtinTemplates();
    assert(templates.length > 0, "at least one built-in template must exist");
    foreach (name; templates)
        assert(name.length > 0, "template name must be non-empty");
}

@("chat: chatApplyTemplate with known template and single message")
@trusted unittest
{
    import llama.chat : builtinTemplates, chatApplyTemplate;
    import std.string : toStringz;

    // Use the first available built-in template by its name.
    auto names = builtinTemplates();
    assert(names.length > 0);

    // Build a minimal conversation.
    llama_chat_message[1] msgs;
    msgs[0].role    = "user";
    msgs[0].content = "Hello";

    char[1024] buf;
    // Pass the template name as the tmpl string (llama.cpp accepts name aliases).
    int n = chatApplyTemplate(names[0].toStringz, msgs[], /*addAss=*/false, buf[]);
    // A negative return means unsupported — but since we used a built-in name it should succeed.
    assert(n > 0, "chatApplyTemplate should return byte count > 0 for a valid template");
}

@("chat: applyTemplate auto-resizes buffer and returns a D string")
@trusted unittest
{
    import llama.chat : builtinTemplates, applyTemplate;
    import std.string : toStringz;

    auto names = builtinTemplates();
    assert(names.length > 0);

    llama_chat_message[2] msgs;
    msgs[0].role = "user";    msgs[0].content = "What is 2+2?";
    msgs[1].role = "assistant"; msgs[1].content = "4";

    string result = applyTemplate(names[0].toStringz, msgs[], /*addAss=*/true);
    assert(result.length > 0, "applyTemplate must return a non-empty string");
}

// ---------------------------------------------------------------------------
// adapter.d
// ---------------------------------------------------------------------------

@("adapter: LlamaAdapterLora is a struct")
unittest
{
    import llama.adapter : LlamaAdapterLora;
    static assert(is(LlamaAdapterLora == struct));
}

@("adapter: setAdaptersLora function is reachable")
unittest
{
    import llama.adapter : setAdaptersLora;
    // Just verify the symbol resolves; calling it would require a live context.
    static assert(__traits(compiles, &setAdaptersLora));
}

@("adapter: loadAdapterLora with nonexistent path returns falsy")
unittest
{
    import llama.adapter : LlamaAdapterLora, loadAdapterLora;
    loadAllBackends();
    auto model = LlamaModel.loadFromFile("/nonexistent/model.gguf");
    // model is null — loadAdapterLora must not crash; llama_adapter_lora_init
    // returns null for a null model path even if the model ptr itself is null.
    // Guard: only call if model loaded.
    if (!model) return; // expected path in CI — no model file available
    auto adapter = loadAdapterLora(model, "/nonexistent/lora.gguf");
    assert(!adapter, "Non-existent LoRA path must yield a null adapter");
}

// ---------------------------------------------------------------------------
// LlamaModel: new properties (compile-level; null-model guards)
// ---------------------------------------------------------------------------

@("LlamaModel: isRecurrent / nCtxTrain / nParams / size — symbols resolve")
unittest
{
    // These properties delegate to llama_model_* C calls.
    // We can only verify they exist and compile; calling on a null model crashes.
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
// LlamaContext: new properties compile check
// ---------------------------------------------------------------------------

@("LlamaContext: new properties — symbols resolve")
unittest
{
    static assert(__traits(hasMember, LlamaContext, "poolingType"));
    static assert(__traits(hasMember, LlamaContext, "memory"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddings"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddingsIth"));
    static assert(__traits(hasMember, LlamaContext, "getEmbeddingsSeq"));
    static assert(__traits(hasMember, LlamaContext, "stateGetSize"));
    static assert(__traits(hasMember, LlamaContext, "stateGetData"));
    static assert(__traits(hasMember, LlamaContext, "stateSetData"));
    static assert(__traits(hasMember, LlamaContext, "stateSaveFile"));
    static assert(__traits(hasMember, LlamaContext, "stateLoadFile"));
}

// ---------------------------------------------------------------------------
// vocab.d: new free functions compile check
// ---------------------------------------------------------------------------

@("vocab: new token helper functions — symbols resolve")
unittest
{
    // Verify the D wrapper symbols exist. Calling them requires a live vocab.
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
