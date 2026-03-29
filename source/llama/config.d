/++
UDA-driven CLI configuration for llama-cpp-d programs.

Annotate struct fields with `@Param` to describe CLI flags.
`parseConfig!T` then uses compile-time metaprogramming to generate
a single `std.getopt` call that covers all annotated fields.

Example:
---
import llama.config;

struct MyConfig {
    @Param("m", "Model path (.gguf file)")
    string modelPath;

    @Param("ngl", "Number of GPU layers (default: 99)")
    int nGpuLayers = 99;

    @Param("n", "Tokens to predict (default: 128)")
    int nPredict = 128;
}

void main(string[] args) {
    MyConfig cfg;
    if (!parseConfig(cfg, args)) return; // --help was printed
    // use cfg.modelPath, cfg.nGpuLayers, cfg.nPredict ...
}
---
+/
module llama.config;

import llama.llama : LLAMA_DEFAULT_SEED;

/// Attach to a field to expose it as a CLI flag.
struct Param
{
    string shortFlag; ///< Short flag name, e.g. `"m"` for `-m`.
    string help; ///< Description shown in `--help` output.
}

// ── Ready-made config structs ────────────────────────────────────────────────

/// Common model + context parameters.
struct ModelConfig
{
    @Param("m", "Model path (.gguf file)")
    string modelPath;

    @Param("ngl", "Number of GPU layers (default: 99)")
    int nGpuLayers = 99;

    @Param("c", "Context size in tokens (0 = model training length)")
    uint nCtx = 0;

    @Param("b", "Batch size in tokens (default: 512)")
    uint nBatch = 512;

    @Param("n", "Tokens to predict (default: 128)")
    int nPredict = 128;

    @Param("p", "Prompt text")
    string prompt = "Hello my name is";
}

/// Sampling hyper-parameters.
struct SamplingConfig
{
    @Param("t", "Temperature; 0.0 = greedy (default: 0.8)")
    float temp = 0.8f;

    @Param("k", "Top-K candidates; 0 = disabled (default: 40)")
    int topK = 40;

    @Param("top-p", "Top-P (nucleus) probability (default: 0.95)")
    float topP = 0.95f;

    @Param("min-p", "Min-P probability floor (default: 0.05)")
    float minP = 0.05f;

    @Param("seed", "RNG seed (default: LLAMA_DEFAULT_SEED)")
    uint seed = LLAMA_DEFAULT_SEED;

    @Param("repeat-penalty", "Repetition penalty; 1.0 = off (default: 1.1)")
    float repeatPenalty = 1.1f;

    @Param("repeat-last-n", "Penalty look-back window; -1 = full context (default: 64)")
    int repeatLastN = 64;
}

/++
Build the getopt argument list for all `@Param`-annotated fields of `T`.

Runs entirely at compile time (CTFE).  The result is a comma-separated
string like:
    `"m|modelPath", "Model path (.gguf file)", &cfg.modelPath,`
which is then `mixin`'d inside `parseConfig`.
+/
private string buildGetoptArgs(T)()
{
    import std.traits : hasUDA, getUDAs, FieldNameTuple;

    string code;
    static foreach (name; FieldNameTuple!T)
    {
        {
            alias field = __traits(getMember, T, name);
            static if (hasUDA!(field, Param))
            {
                enum uda = getUDAs!(field, Param)[0];
                code ~= `"` ~ uda.shortFlag ~ `|` ~ name ~ `", `
                    ~ `"` ~ uda.help ~ `", `
                    ~ `&cfg.` ~ name ~ `, `;
            }
        }
    }
    return code;
}

/++
Parse `args` into `cfg` using the `@Param` UDAs on `T`'s fields.

Recognised flags are removed from `args` in-place (std.getopt semantics).
If `--help` or `-h` is present the usage is printed and `false` is returned.
Returns `true` on success.

Multiple config structs can be parsed sequentially; each call only removes
its own flags.
+/
bool parseConfig(T)(ref T cfg, ref string[] args, string banner = "Options:")
{
    import std.getopt : getopt, defaultGetoptPrinter, config;

    auto r = mixin("getopt(args, config.passThrough, " ~ buildGetoptArgs!T() ~ ")");
    if (r.helpWanted)
    {
        defaultGetoptPrinter(banner, r.options);
        return false;
    }
    return true;
}

/++
Build a `SamplerChain` from a `SamplingConfig`.

The chain applies (in order): penalties → temperature → top-K → top-P →
min-P → dist.  Greedy mode is used when `cfg.temp <= 0`.
+/
auto buildSamplerChain(ref const SamplingConfig cfg) @trusted
{
    import llama.sampling : SamplerChain;

    auto smpl = SamplerChain.create();
    if (cfg.repeatPenalty != 1.0f || cfg.repeatLastN != 0)
        smpl.penalties(cfg.repeatLastN, cfg.repeatPenalty);
    if (cfg.temp <= 0.0f)
    {
        smpl.greedy();
    }
    else
    {
        smpl.temp(cfg.temp);
        if (cfg.topK > 0)
            smpl.topK(cfg.topK);
        if (cfg.topP < 1.0f)
            smpl.topP(cfg.topP);
        if (cfg.minP > 0.0f)
            smpl.minP(cfg.minP);
        smpl.dist(cfg.seed);
    }
    return smpl;
}
