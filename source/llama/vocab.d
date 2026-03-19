module llama.vocab;

import llama.llama;

// Drop const for C APIs that don't take const vocab pointer.
private llama_vocab* mutableVocab(const(llama_vocab)* vocab) @trusted @nogc nothrow
{
    return cast(llama_vocab*) vocab;
}

/// Split `text` into tokens. Returns a GC-allocated slice.
llama_token[] tokenize(const(llama_vocab)* vocab, const(char)[] text,
                       bool addSpecial = true, bool parseSpecial = true) @trusted
{
    if (text.length == 0) return null;
    auto v = mutableVocab(vocab);
    int n = -llama_tokenize(v, text.ptr, cast(int) text.length, null, 0, addSpecial, parseSpecial);
    if (n <= 0) return null;
    auto tokens = new llama_token[](n);
    int result = llama_tokenize(v, text.ptr, cast(int) text.length,
                                tokens.ptr, cast(int) tokens.length, addSpecial, parseSpecial);
    return result < 0 ? null : tokens[0 .. result];
}

/// The string piece for a single token.
string tokenToString(const(llama_vocab)* vocab, llama_token token) @trusted
{
    char[256] buf;
    int n = llama_token_to_piece(mutableVocab(vocab), token, buf.ptr, cast(int) buf.length, 0, true);
    return n < 0 ? null : buf[0 .. n].idup;
}

/// Decode a token sequence back into text.
string detokenize(const(llama_vocab)* vocab, const(llama_token)[] tokens,
                  bool removeSpecial = false, bool unparseSpecial = false) @trusted
{
    if (tokens.length == 0) return "";
    auto v = mutableVocab(vocab);
    int bufSize = cast(int)(tokens.length * 8 + 64);
    char[] buf = new char[](bufSize);
    int n = llama_detokenize(v, tokens.ptr, cast(int) tokens.length,
                             buf.ptr, cast(int) buf.length, removeSpecial, unparseSpecial);
    if (n < 0)
    {
        buf = new char[](-n + 1);
        n = llama_detokenize(v, tokens.ptr, cast(int) tokens.length,
                             buf.ptr, cast(int) buf.length, removeSpecial, unparseSpecial);
        if (n < 0) return "";
    }
    return buf[0 .. n].idup;
}

llama_token bosToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_bos(mutableVocab(vocab)); } /// BOS token.
llama_token eosToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_eos(mutableVocab(vocab)); } /// EOS token.
llama_token eotToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_eot(mutableVocab(vocab)); } /// EOT (end-of-turn) token.
llama_token nlToken (const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_nl (mutableVocab(vocab)); } /// Newline token.
llama_token padToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_pad(mutableVocab(vocab)); } /// Padding token.
llama_token sepToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_sep(mutableVocab(vocab)); } /// Sentence separator token.

// Fill-in-the-Middle special tokens.
llama_token fimPreToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_pre(mutableVocab(vocab)); } /// FIM prefix token.
llama_token fimSufToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_suf(mutableVocab(vocab)); } /// FIM suffix token.
llama_token fimMidToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_mid(mutableVocab(vocab)); } /// FIM middle token.
llama_token fimPadToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_pad(mutableVocab(vocab)); } /// FIM padding token.
llama_token fimRepToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_rep(mutableVocab(vocab)); } /// FIM repo token.
llama_token fimSepToken(const(llama_vocab)* vocab) @trusted @nogc nothrow { return llama_vocab_fim_sep(mutableVocab(vocab)); } /// FIM separator token.

/// Vocabulary type as int (compare to `LLAMA_VOCAB_TYPE_*` constants).
int vocabType(const(llama_vocab)* vocab) @trusted @nogc nothrow
{
    return cast(int) llama_vocab_type(mutableVocab(vocab));
}

/// Raw text piece for a token (pointer into model memory; do not free).
const(char)* tokenText(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_get_text(mutableVocab(vocab), token);
}

/// Log-probability score stored for a token in the vocab.
float tokenScore(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_get_score(mutableVocab(vocab), token);
}

/// Token attribute flags (control, normal, byte, etc.).
llama_token_attr tokenAttr(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_get_attr(mutableVocab(vocab), token);
}

/// True if the token is a control token (not renderable text).
bool isControl(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_is_control(mutableVocab(vocab), token);
}

/// True if the token signals end of generation.
bool isEog(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_is_eog(mutableVocab(vocab), token);
}
