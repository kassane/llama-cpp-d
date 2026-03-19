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

/// True if the token signals end of generation.
bool isEog(const(llama_vocab)* vocab, llama_token token) @trusted @nogc nothrow
{
    return llama_vocab_is_eog(mutableVocab(vocab), token);
}
