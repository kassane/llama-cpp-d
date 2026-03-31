/++
Retrieval-Augmented Generation (RAG) helpers.

Provides an in-memory vector store backed by cosine similarity, and utilities
for chunking text, embedding documents, and building RAG prompts.

Usage pattern:
---
import llama, llama.rag;

// 1. Embed documents
auto store = VectorStore();
store.addDocument("doc-1", "D is a systems programming language.", embeddingOf(model, ctx, vocab, "..."));

// 2. At query time, retrieve top-K chunks
float[] qEmb = embeddingOf(model, ctx, vocab, userQuery);
auto hits = store.retrieve(qEmb, 3);

// 3. Build the augmented prompt and feed it to the LLM
string prompt = buildRagPrompt(hits, userQuery);
---
+/
module llama.rag;

import std.math : sqrt;

// ── Core types ───────────────────────────────────────────────────────────────

/// A stored text chunk together with its L2-normalised embedding.
struct Document
{
    string id; ///< Caller-supplied identifier.
    string text; ///< Original chunk text.
    float[] embedding; ///< L2-normalised embedding vector.
}

/// Retrieval result: a matching document and its similarity score.
struct Hit
{
    string id;
    string text;
    float score; ///< Cosine similarity in [-1, 1]; higher is more similar.
}

// ── Vector store ─────────────────────────────────────────────────────────────

/++
In-memory vector store.

Documents are stored with L2-normalised embeddings so that inner-product
equals cosine similarity (O(N * D) linear scan; sufficient for thousands of
chunks).
+/
struct VectorStore
{
    private Document[] _docs;

    /// Number of stored documents.
    @property size_t length() const @nogc nothrow
    {
        return _docs.length;
    }

    /++
    Add a document with its embedding.
    `embedding` is copied and L2-normalised internally.
    +/
    void addDocument(string id, string text, scope const(float)[] embedding)
    {
        _docs ~= Document(id, text, l2Normalize(embedding));
    }

    /++
    Retrieve the `topK` most similar documents for `query`.
    `query` need not be pre-normalised; it is normalised here.
    Returns hits sorted by descending similarity.
    +/
    Hit[] retrieve(scope const(float)[] query, int topK = 3) @trusted const
    {
        if (_docs.length == 0 || query.length == 0)
            return null;
        float[] qn = l2Normalize(query);

        // Score all documents.
        auto scores = new float[](_docs.length);
        foreach (i, ref doc; _docs)
            scores[i] = dot(qn, doc.embedding);

        // Manual insertion sort on indices: avoids std.algorithm.sort's move()
        // which triggers a compile-time moveEmplace safety check under -preview=safer.
        auto idx = new size_t[](_docs.length);
        foreach (i; 0 .. idx.length)
            idx[i] = i;
        for (size_t i = 1; i < idx.length; i++)
        {
            size_t key = idx[i];
            float ks = scores[key];
            size_t j = i;
            while (j > 0 && scores[idx[j - 1]] < ks)
            {
                idx[j] = idx[j - 1];
                j--;
            }
            idx[j] = key;
        }

        size_t n = (topK >= 0 && topK < cast(int) idx.length)
            ? cast(size_t) topK : idx.length;
        auto hits = new Hit[](n);
        foreach (i; 0 .. n)
            hits[i] = Hit(_docs[idx[i]].id, _docs[idx[i]].text, scores[idx[i]]);
        return hits;
    }

    /// Remove all stored documents.
    void clear()
    {
        _docs = null;
    }
}

// ── Prompt assembly ───────────────────────────────────────────────────────────

/++
Build a RAG-augmented prompt from retrieved chunks and the user query.

The resulting string follows a simple template:
    Context:
    [1] <chunk text>
    [2] <chunk text>
    ...
    Question: <userQuery>
    Answer:
+/
string buildRagPrompt(scope const Hit[] hits, string userQuery)
{
    import std.array : Appender;

    Appender!string buf;
    buf.reserve(512 + hits.length * 256);

    buf ~= "Context:\n";
    foreach (i, ref h; hits)
    {
        import std.conv : to;

        buf ~= "[";
        buf ~= (i + 1).to!string;
        buf ~= "] ";
        buf ~= h.text;
        buf ~= "\n";
    }
    buf ~= "\nQuestion: ";
    buf ~= userQuery;
    buf ~= "\nAnswer:";
    return buf.data;
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Cosine similarity of two vectors (need not be pre-normalised).
float cosineSimilarity(scope const(float)[] a, scope const(float)[] b) @nogc nothrow
in (a.length == b.length && a.length > 0)
{
    float na = 0, nb = 0, ab = 0;
    foreach (i; 0 .. a.length)
    {
        na += a[i] * a[i];
        nb += b[i] * b[i];
        ab += a[i] * b[i];
    }
    float denom = sqrt(na) * sqrt(nb);
    return denom > 1e-9f ? ab / denom : 0.0f;
}

// Private helpers

private float[] l2Normalize(scope const(float)[] v)
{
    float norm = 0;
    foreach (x; v)
        norm += x * x;
    norm = sqrt(norm);
    auto out_ = new float[](v.length);
    if (norm > 1e-9f)
        foreach (i; 0 .. v.length)
            out_[i] = v[i] / norm;
    else
        out_[] = v[];
    return out_;
}

private float dot(scope const(float)[] a, scope const(float)[] b) @nogc nothrow
{
    float s = 0;
    foreach (i; 0 .. a.length)
        s += a[i] * b[i];
    return s;
}
