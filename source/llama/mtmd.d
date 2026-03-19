/++
D bindings and RAII wrappers for libmtmd (multimodal support).

libmtmd encodes images and audio into token embeddings that a language
model can attend to alongside ordinary text tokens.

Typical usage:
---
auto mtmd = MtmdContext.initFromFile("mmproj.gguf", model.ptr);
auto bmp  = mtmd.loadBitmap("photo.jpg");
auto chunks = InputChunks.create();
auto txt  = mtmd_input_text("<__media__>\nDescribe this image.".ptr, true, true);
mtmd.tokenize(chunks, txt, [bmp.ptr]);
llama_pos nPast;
mtmd.evalChunks(ctx.ptr, chunks, 0, 0, 512, true, nPast);
// ... then sample as usual with SamplerChain
---
+/
module llama.mtmd;

import llama.llama; // llama_model, llama_context, llama_token, llama_pos, llama_seq_id

// ── Callback aliases (matching ggml.h / ggml-backend.h) ────────────────────
/// Logging callback: `level` is `ggml_log_level` cast to int.
alias ggml_log_callback = extern(C) void function(int level, const(char)* text, void* user_data) nothrow;
/// Eval callback: return false to cancel.
alias ggml_backend_sched_eval_callback = extern(C) bool function(void* tensor, bool ask, void* user_data) nothrow;

// ── Opaque C types ──────────────────────────────────────────────────────────
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;

// ── Plain C structs ─────────────────────────────────────────────────────────

/// Chunk type emitted by `mtmd_tokenize`.
enum mtmd_input_chunk_type : int
{
    text  = 0,
    image = 1,
    audio = 2,
}

/// Text input descriptor passed to `mtmd_tokenize`.
struct mtmd_input_text
{
    const(char)* text;        ///< Prompt, may contain `mtmd_default_marker()`.
    bool         add_special;  ///< Prepend BOS token if true.
    bool         parse_special;///< Interpret special tokens in the prompt.
}

/++
Parameters for `mtmd_init_from_file`.
Always initialise via `mtmd_context_params_default()` and then override fields.
+/
struct mtmd_context_params
{
    bool  use_gpu;                           ///< Run the projector on GPU if available.
    bool  print_timings;                     ///< Print encoder timing stats on free.
    int   n_threads;                         ///< Number of CPU threads (0 = auto).
    const(char)* image_marker;               ///< Deprecated; use `media_marker`.
    const(char)* media_marker;               ///< Marker replaced by image/audio tokens (default: `<__media__>`).
    int   flash_attn_type;                   ///< `llama_flash_attn_type` cast to int.
    bool  warmup;                            ///< Run a warm-up encode pass after init.
    int   image_min_tokens;                  ///< Dynamic-resolution lower bound (0 = from metadata).
    int   image_max_tokens;                  ///< Dynamic-resolution upper bound (0 = from metadata).
    ggml_backend_sched_eval_callback cb_eval;///< Optional eval callback.
    void* cb_eval_user_data;                 ///< User data for `cb_eval`.
}

// ── C API declarations ──────────────────────────────────────────────────────
extern(C) @nogc nothrow:

/// Returns the default media marker string (`"<__media__>"`).
const(char)* mtmd_default_marker();

/// Returns a default-initialised `mtmd_context_params`.
mtmd_context_params mtmd_context_params_default();

/++
Initialises a multimodal context from a projector GGUF file.
Returns `null` on failure (bad path, incompatible model, etc.).
+/
mtmd_context* mtmd_init_from_file(
    const(char)* mmproj_fname,
    const(llama_model)* text_model,
    mtmd_context_params ctx_params);

/// Frees a multimodal context.
void mtmd_free(mtmd_context* ctx);

/// True if the model requires a non-causal attention mask for llama_decode.
bool mtmd_decode_use_non_causal(mtmd_context* ctx);

/// True if the model uses M-RoPE (Multimodal RoPE) for llama_decode.
bool mtmd_decode_use_mrope(mtmd_context* ctx);

/// True if the model supports image input.
bool mtmd_support_vision(mtmd_context* ctx);

/// True if the model supports audio input.
bool mtmd_support_audio(mtmd_context* ctx);

/// Audio sample rate in Hz (e.g. 16 000 for Whisper), or -1 if unsupported.
int mtmd_get_audio_sample_rate(mtmd_context* ctx);

// ── mtmd_bitmap ─────────────────────────────────────────────────────────────

/// Create an image bitmap from raw RGB pixels (`RGBRGBRGB…`; length must equal `nx * ny * 3`).
mtmd_bitmap* mtmd_bitmap_init(uint nx, uint ny, const(ubyte)* data);

/// Create an audio bitmap from float PCM samples.
mtmd_bitmap* mtmd_bitmap_init_from_audio(size_t n_samples, const(float)* data);

uint          mtmd_bitmap_get_nx    (const(mtmd_bitmap)* bitmap);
uint          mtmd_bitmap_get_ny    (const(mtmd_bitmap)* bitmap);
const(ubyte)* mtmd_bitmap_get_data  (const(mtmd_bitmap)* bitmap);
size_t        mtmd_bitmap_get_n_bytes(const(mtmd_bitmap)* bitmap);
bool          mtmd_bitmap_is_audio  (const(mtmd_bitmap)* bitmap);
void          mtmd_bitmap_free      (mtmd_bitmap* bitmap);

/// Optional string ID used for KV-cache tracking.
const(char)*  mtmd_bitmap_get_id(const(mtmd_bitmap)* bitmap);
/// Set the bitmap ID.
void          mtmd_bitmap_set_id(mtmd_bitmap* bitmap, const(char)* id);

// ── mtmd_input_chunks ───────────────────────────────────────────────────────

mtmd_input_chunks*       mtmd_input_chunks_init();
size_t                   mtmd_input_chunks_size(const(mtmd_input_chunks)* chunks);
const(mtmd_input_chunk)* mtmd_input_chunks_get (const(mtmd_input_chunks)* chunks, size_t idx);
void                     mtmd_input_chunks_free(mtmd_input_chunks* chunks);

// ── mtmd_input_chunk ────────────────────────────────────────────────────────

mtmd_input_chunk_type     mtmd_input_chunk_get_type        (const(mtmd_input_chunk)* chunk);
const(llama_token)*       mtmd_input_chunk_get_tokens_text (const(mtmd_input_chunk)* chunk, size_t* n_tokens_output);
const(mtmd_image_tokens)* mtmd_input_chunk_get_tokens_image(const(mtmd_input_chunk)* chunk);
size_t                    mtmd_input_chunk_get_n_tokens    (const(mtmd_input_chunk)* chunk);
const(char)*              mtmd_input_chunk_get_id          (const(mtmd_input_chunk)* chunk);
llama_pos                 mtmd_input_chunk_get_n_pos       (const(mtmd_input_chunk)* chunk);
mtmd_input_chunk*         mtmd_input_chunk_copy            (const(mtmd_input_chunk)* chunk);
void                      mtmd_input_chunk_free            (mtmd_input_chunk* chunk);

// ── mtmd_image_tokens ───────────────────────────────────────────────────────

size_t       mtmd_image_tokens_get_n_tokens(const(mtmd_image_tokens)* image_tokens);
size_t       mtmd_image_tokens_get_nx      (const(mtmd_image_tokens)* image_tokens);
size_t       mtmd_image_tokens_get_ny      (const(mtmd_image_tokens)* image_tokens);
const(char)* mtmd_image_tokens_get_id      (const(mtmd_image_tokens)* image_tokens);
llama_pos    mtmd_image_tokens_get_n_pos   (const(mtmd_image_tokens)* image_tokens);

// ── Tokenise / encode ───────────────────────────────────────────────────────

/++
Tokenise a text prompt that may contain media markers.
Returns 0 on success, 1 on bitmap-count mismatch, 2 on preprocessing error.
+/
int mtmd_tokenize(
    mtmd_context*       ctx,
    mtmd_input_chunks*  output,
    const(mtmd_input_text)* text,
    const(mtmd_bitmap*)* bitmaps,
    size_t              n_bitmaps);

/// Encode a single image/audio chunk.  Returns 0 on success.
int mtmd_encode_chunk(mtmd_context* ctx, const(mtmd_input_chunk)* chunk);

/// Pointer to the float embeddings from the last `mtmd_encode_chunk` call.
float* mtmd_get_output_embd(mtmd_context* ctx);

/// Set a logging callback.
void mtmd_log_set(ggml_log_callback log_callback, void* user_data);

// ── mtmd-helper API ─────────────────────────────────────────────────────────

/// Set logging callback (also calls `mtmd_log_set` internally).
void mtmd_helper_log_set(ggml_log_callback log_callback, void* user_data);

/// Load an image or audio file into a bitmap.  Thread-safe.  Returns `null` on failure.
mtmd_bitmap* mtmd_helper_bitmap_init_from_file(mtmd_context* ctx, const(char)* fname);

/// Load from an in-memory buffer (JPEG/PNG/BMP/GIF/WAV/MP3/FLAC).  Thread-safe.
mtmd_bitmap* mtmd_helper_bitmap_init_from_buf(mtmd_context* ctx, const(ubyte)* buf, size_t len);

/// Total token count across all chunks (for KV-cache sizing).
size_t    mtmd_helper_get_n_tokens(const(mtmd_input_chunks)* chunks);

/// Total position count across all chunks (may differ from n_tokens for M-RoPE).
llama_pos mtmd_helper_get_n_pos(const(mtmd_input_chunks)* chunks);

/++
Eval all chunks: text via `llama_decode`, images via `mtmd_encode_chunk` + `llama_decode`.
Returns 0 on success. NOT thread-safe.
+/
int mtmd_helper_eval_chunks(
    mtmd_context*             ctx,
    llama_context*            lctx,
    const(mtmd_input_chunks)* chunks,
    llama_pos                 n_past,
    llama_seq_id              seq_id,
    int                       n_batch,
    bool                      logits_last,
    llama_pos*                new_n_past);

/// Like `mtmd_helper_eval_chunks` but for a single chunk.
int mtmd_helper_eval_chunk_single(
    mtmd_context*            ctx,
    llama_context*           lctx,
    const(mtmd_input_chunk)* chunk,
    llama_pos                n_past,
    llama_seq_id             seq_id,
    int                      n_batch,
    bool                     logits_last,
    llama_pos*               new_n_past);

/// Decode an already-encoded image chunk (embeddings pre-calculated).
int mtmd_helper_decode_image_chunk(
    mtmd_context*            ctx,
    llama_context*           lctx,
    const(mtmd_input_chunk)* chunk,
    float*                   encoded_embd,
    llama_pos                n_past,
    llama_seq_id             seq_id,
    int                      n_batch,
    llama_pos*               new_n_past);

// ── D wrappers ──────────────────────────────────────────────────────────────
// Reset to D linkage — the extern(C) block above must not bleed through.
extern(D):

/++
RAII owner of a `mtmd_bitmap*`.

Construct via `MtmdBitmap.fromRGB`, `MtmdBitmap.fromAudio`, or
`MtmdContext.loadBitmap`. Not copyable; move via `std.algorithm.move`.
+/
struct MtmdBitmap
{
    private mtmd_bitmap* _bmp;

    @disable this();
    @disable this(this);

    private this(mtmd_bitmap* b) @nogc nothrow { _bmp = b; }

    ~this() @nogc nothrow
    {
        if (_bmp) { mtmd_bitmap_free(_bmp); _bmp = null; }
    }

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

    bool opCast(T : bool)() const @nogc nothrow { return _bmp !is null; }

    @property uint          nx()      const @nogc nothrow { return mtmd_bitmap_get_nx(_bmp); }
    @property uint          ny()      const @nogc nothrow { return mtmd_bitmap_get_ny(_bmp); }
    @property bool          isAudio() const @nogc nothrow { return mtmd_bitmap_is_audio(_bmp); }
    @property mtmd_bitmap*  ptr()           @nogc nothrow { return _bmp; }

    /// Raw pixel/sample bytes (read-only slice into C memory).
    @property const(ubyte)[] data() const @trusted @nogc nothrow
    {
        return mtmd_bitmap_get_data(_bmp)[0 .. mtmd_bitmap_get_n_bytes(_bmp)];
    }

    /// Optional KV-cache tracking ID.
    const(char)* id()              const @nogc nothrow { return mtmd_bitmap_get_id(_bmp); }
    void         setId(const(char)* s)   @nogc nothrow { mtmd_bitmap_set_id(_bmp, s); }
}

// ────────────────────────────────────────────────────────────────────────────

/++
RAII owner of a `mtmd_input_chunks*` list.
Populated by `MtmdContext.tokenize`.
Supports `foreach` iteration over `const(mtmd_input_chunk)*` elements.
+/
struct InputChunks
{
    private mtmd_input_chunks* _chunks;

    @disable this();
    @disable this(this);

    private this(mtmd_input_chunks* c) @nogc nothrow { _chunks = c; }

    ~this() @nogc nothrow
    {
        if (_chunks) { mtmd_input_chunks_free(_chunks); _chunks = null; }
    }

    /// Create an empty chunk list (to be filled by `MtmdContext.tokenize`).
    static InputChunks create() @nogc nothrow
    {
        return InputChunks(mtmd_input_chunks_init());
    }

    /// Number of chunks.
    @property size_t length() const @nogc nothrow { return mtmd_input_chunks_size(_chunks); }
    /// True when no chunks are present.
    @property bool   empty()  const @nogc nothrow { return length == 0; }

    /// Index into the chunk list.
    const(mtmd_input_chunk)* opIndex(size_t idx) const @nogc nothrow
    {
        return mtmd_input_chunks_get(_chunks, idx);
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

    /// Raw C pointer (passed to helper functions).
    @property mtmd_input_chunks*       ptr()       @nogc nothrow { return _chunks; }
    @property const(mtmd_input_chunks)* ptr() const @nogc nothrow { return _chunks; }

    /// Total token count across all chunks.
    @property size_t    nTokens() const @nogc nothrow { return mtmd_helper_get_n_tokens(_chunks); }
    /// Total position count (may differ from `nTokens` for M-RoPE models).
    @property llama_pos nPos()    const @nogc nothrow { return mtmd_helper_get_n_pos(_chunks); }
}

// ────────────────────────────────────────────────────────────────────────────

/++
RAII owner of a `mtmd_context*` (multimodal projector).
Initialise via `MtmdContext.initFromFile`. Check `if (ctx)` after construction.
+/
struct MtmdContext
{
    private mtmd_context* _ctx;

    @disable this();
    @disable this(this);

    private this(mtmd_context* c) @nogc nothrow { _ctx = c; }

    ~this() @nogc nothrow
    {
        if (_ctx) { mtmd_free(_ctx); _ctx = null; }
    }

    /// Load a projector from a GGUF file. Returns a falsy context on failure or null model.
    static MtmdContext initFromFile(
        string mmproj,
        const(llama_model)* model,
        mtmd_context_params params) @trusted nothrow
    {
        if (model is null) return MtmdContext(null);
        import std.string : toStringz;
        return MtmdContext(mtmd_init_from_file(mmproj.toStringz, model, params));
    }

    /// Overload using default params.
    static MtmdContext initFromFile(string mmproj, const(llama_model)* model) nothrow
    {
        auto p = mtmd_context_params_default();
        return initFromFile(mmproj, model, p);
    }

    bool opCast(T : bool)() const @nogc nothrow { return _ctx !is null; }

    @property mtmd_context* ptr()         @nogc nothrow { return _ctx; }
    @property bool supportsVision()       @nogc nothrow { return mtmd_support_vision(_ctx); }
    @property bool supportsAudio()        @nogc nothrow { return mtmd_support_audio(_ctx); }
    @property bool useNonCausal()         @nogc nothrow { return mtmd_decode_use_non_causal(_ctx); }
    @property bool useMrope()             @nogc nothrow { return mtmd_decode_use_mrope(_ctx); }
    @property int  audioSampleRate()      @nogc nothrow { return mtmd_get_audio_sample_rate(_ctx); }

    /// Load an image or audio file into an owned bitmap.  Returns falsy bitmap on failure.
    MtmdBitmap loadBitmap(string path) @trusted nothrow
    {
        import std.string : toStringz;
        return MtmdBitmap(mtmd_helper_bitmap_init_from_file(_ctx, path.toStringz));
    }

    /// Load a bitmap from an in-memory byte buffer.
    MtmdBitmap loadBitmapFromBuf(const(ubyte)[] buf) @trusted @nogc nothrow
    {
        return MtmdBitmap(mtmd_helper_bitmap_init_from_buf(_ctx, buf.ptr, buf.length));
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
        return mtmd_tokenize(_ctx, output.ptr, &text, bitmaps.ptr, bitmaps.length);
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
            _ctx, lctx, chunks.ptr,
            nPast, seqId, nBatch, logitsLast, &newNPast);
    }

    /// Encode a single image chunk and return its embeddings.
    /// Returns 0 on success; the embedding pointer is valid until the next encode.
    int encodeChunk(const(mtmd_input_chunk)* chunk) @nogc nothrow
    {
        return mtmd_encode_chunk(_ctx, chunk);
    }

    /// Pointer to the most recently encoded embeddings.
    float* outputEmbd() @nogc nothrow { return mtmd_get_output_embd(_ctx); }
}
