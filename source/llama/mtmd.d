/++
D bindings and wrappers for libmtmd (multimodal support).

libmtmd encodes images and audio into token embeddings that a language
model can attend to alongside ordinary text tokens.

Typical usage:
---
auto mtmd = MtmdContext.initFromFile("mmproj.gguf", model.ptr);
auto bmp  = mtmd.loadBitmap("photo.jpg");
auto chunks = InputChunks.create();
auto txt  = mtmd_input_text(&fullPrompt[0], true, true);
mtmd.tokenize(chunks, txt, [bmp.ptr]);
llama_pos nPast;
mtmd.evalChunks(ctx.ptr, chunks, 0, 0, 512, true, nPast);
// ... then sample as usual with SamplerChain
---
+/
module llama.mtmd;

import llama.llama; // llama_model, llama_context, llama_token, llama_pos, llama_seq_id
import llama.owned;

// All C declarations (mtmd_*, ggml_log_callback, ggml_backend_sched_eval_callback,
// llama_flash_attn_type, etc.) are auto-imported from the mtmd headers via importC.
// Private: mtmd.h includes llama.h, so public re-export would duplicate llama_stubs symbols.
private import c.mtmd_stubs;

// ── D wrappers ───────────────────────────────────────────────────────────────

/++
An image or audio bitmap loaded from a file or raw buffer.
Construct via `MtmdBitmap.fromRGB`, `MtmdBitmap.fromAudio`, or
`MtmdContext.loadBitmap`.
+/
struct MtmdBitmap
{
    mixin Owned!(mtmd_bitmap, mtmd_bitmap_free);

    /++ Create from a raw RGB pixel buffer (`RGBRGBRGB…`). `data.length` must equal `nx * ny * 3`. +/
    static MtmdBitmap fromRGB(uint nx, uint ny, scope const(ubyte)[] data) @trusted @nogc nothrow
    in (data.length == cast(size_t) nx * ny * 3)
    {
        return MtmdBitmap(mtmd_bitmap_init(nx, ny, data.ptr));
    }

    /// Create from float PCM audio samples (mono, any sample rate).
    static MtmdBitmap fromAudio(scope const(float)[] samples) @trusted @nogc nothrow
    {
        return MtmdBitmap(mtmd_bitmap_init_from_audio(samples.length, samples.ptr));
    }

    // importC drops `const` from C headers, so const methods need @trusted + cast.
    @property uint nx()      const @trusted @nogc nothrow { return mtmd_bitmap_get_nx     (cast(mtmd_bitmap*) _ptr); }
    @property uint ny()      const @trusted @nogc nothrow { return mtmd_bitmap_get_ny     (cast(mtmd_bitmap*) _ptr); }
    @property bool isAudio() const @trusted @nogc nothrow { return mtmd_bitmap_is_audio   (cast(mtmd_bitmap*) _ptr); }

    /// Raw pixel/sample bytes (read-only slice into C memory).
    @property const(ubyte)[] data() const @trusted @nogc nothrow
    {
        auto p = cast(mtmd_bitmap*) _ptr;
        return mtmd_bitmap_get_data(p)[0 .. mtmd_bitmap_get_n_bytes(p)];
    }

    /// Optional KV-cache tracking ID.
    const(char)* id()              const @trusted @nogc nothrow { return mtmd_bitmap_get_id (cast(mtmd_bitmap*) _ptr); }
    void         setId(const(char)* s)         @nogc nothrow { mtmd_bitmap_set_id(_ptr, s); }
}

// ────────────────────────────────────────────────────────────────────────────

/++
A list of tokenized input chunks produced by `MtmdContext.tokenize`.
Supports `foreach` iteration over `const(mtmd_input_chunk)*` elements.
+/
struct InputChunks
{
    mixin Owned!(mtmd_input_chunks, mtmd_input_chunks_free);

    /// Create an empty chunk list (to be filled by `MtmdContext.tokenize`).
    static InputChunks create() @nogc nothrow
    {
        return InputChunks(mtmd_input_chunks_init());
    }

    /// Number of chunks.
    @property size_t length() const @trusted @nogc nothrow { return mtmd_input_chunks_size(cast(mtmd_input_chunks*) _ptr); }
    /// True when no chunks are present.
    @property bool   empty()  const @nogc nothrow { return length == 0; }

    /// Index into the chunk list.
    const(mtmd_input_chunk)* opIndex(size_t idx) const @trusted @nogc nothrow
    {
        return mtmd_input_chunks_get(cast(mtmd_input_chunks*) _ptr, idx);
    }

    /// `foreach (chunk; chunks)` — iterates each `const(mtmd_input_chunk)*`.
    int opApply(scope int delegate(const(mtmd_input_chunk)*) dg) const
    {
        foreach (i; 0 .. length)
            if (auto r = dg(this[i])) return r;
        return 0;
    }

    /// `foreach (i, chunk; chunks)` — indexed iteration.
    int opApply(scope int delegate(size_t, const(mtmd_input_chunk)*) dg) const
    {
        foreach (i; 0 .. length)
            if (auto r = dg(i, this[i])) return r;
        return 0;
    }

    /// Total token count across all chunks.
    @property size_t    nTokens() const @trusted @nogc nothrow { return mtmd_helper_get_n_tokens(cast(mtmd_input_chunks*) _ptr); }
    /// Total position count (may differ from `nTokens` for M-RoPE models).
    @property llama_pos nPos()    const @trusted @nogc nothrow { return mtmd_helper_get_n_pos   (cast(mtmd_input_chunks*) _ptr); }
}

// ────────────────────────────────────────────────────────────────────────────

/++
A multimodal projector context loaded from a GGUF file.
Encodes images and audio into embeddings for the paired language model.
Check `if (ctx)` after construction.
+/
struct MtmdContext
{
    mixin Owned!(mtmd_context, mtmd_free);

    /// Load a projector from a GGUF file. Returns a falsy context on failure or null model.
    static MtmdContext initFromFile(
        string mmproj,
        const(llama_model)* model,
        mtmd_context_params params) @trusted nothrow
    {
        if (model is null) return MtmdContext(null);
        import std.string : toStringz;
        return MtmdContext(mtmd_init_from_file(mmproj.toStringz, cast(llama_model*) model, params));
    }

    /// Overload using default params.
    static MtmdContext initFromFile(string mmproj, const(llama_model)* model) nothrow
    {
        auto p = mtmd_context_params_default();
        return initFromFile(mmproj, model, p);
    }

    @property bool supportsVision() @nogc nothrow { return mtmd_support_vision(_ptr); }
    @property bool supportsAudio()  @nogc nothrow { return mtmd_support_audio(_ptr); }
    @property bool useNonCausal()   @nogc nothrow { return mtmd_decode_use_non_causal(_ptr); }
    @property bool useMrope()       @nogc nothrow { return mtmd_decode_use_mrope(_ptr); }
    @property int  audioSampleRate()@nogc nothrow { return mtmd_get_audio_sample_rate(_ptr); }

    /// Load an image or audio file into an owned bitmap. Returns falsy bitmap on failure.
    MtmdBitmap loadBitmap(string path) @trusted nothrow
    {
        import std.string : toStringz;
        return MtmdBitmap(mtmd_helper_bitmap_init_from_file(_ptr, path.toStringz));
    }

    /// Load a bitmap from an in-memory byte buffer.
    MtmdBitmap loadBitmapFromBuf(const(ubyte)[] buf) @trusted @nogc nothrow
    {
        return MtmdBitmap(mtmd_helper_bitmap_init_from_buf(_ptr, buf.ptr, buf.length));
    }

    /++
    Tokenise a prompt string that contains `mtmd_default_marker()` placeholders.
    `bitmaps` must have exactly as many entries as markers in `text.text`.
    Returns 0 on success, 1 on count mismatch, 2 on preprocessing error.
    +/
    int tokenize(
        ref InputChunks           output,
        scope ref mtmd_input_text text,
        const(mtmd_bitmap*)[]     bitmaps = null) @trusted @nogc nothrow
    {
        return mtmd_tokenize(_ptr, output.ptr, &text, cast(mtmd_bitmap**) bitmaps.ptr, bitmaps.length);
    }

    /++
    Evaluate all chunks against the language-model context.
    Advances `newNPast` to the position after the last evaluated token.
    Returns 0 on success.
    +/
    int evalChunks(
        llama_context*        lctx,
        ref const InputChunks chunks,
        llama_pos             nPast,
        llama_seq_id          seqId,
        int                   nBatch,
        bool                  logitsLast,
        ref llama_pos         newNPast) @trusted @nogc nothrow
    {
        return mtmd_helper_eval_chunks(
            _ptr, lctx, cast(mtmd_input_chunks*) chunks.ptr,
            nPast, seqId, nBatch, logitsLast, &newNPast);
    }

    /// Encode a single image chunk. Returns 0 on success; the embedding pointer
    /// is valid until the next encode call.
    int encodeChunk(const(mtmd_input_chunk)* chunk) @trusted @nogc nothrow
    {
        return mtmd_encode_chunk(_ptr, cast(mtmd_input_chunk*) chunk);
    }

    /// Pointer to the most recently encoded embeddings.
    float* outputEmbd() @nogc nothrow { return mtmd_get_output_embd(_ptr); }
}
