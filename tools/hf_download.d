/++
Download GGUF model files from HuggingFace Hub.

Without `-f`, lists all `.gguf` files in the repository.
With `-f`, downloads the specified file via curl with a progress bar.

Authentication via `-t TOKEN` or the `HF_TOKEN` environment variable
(required for private repositories and to raise rate limits).

Usage:
  hf-download -r owner/repo [-f filename] [-o outdir] [-t token]
  HF_TOKEN=<token> hf-download -r owner/repo

Requires curl to be installed and available in PATH.

Compatibility note (-preview=all):
  std.json is excluded because JSONValue.toString() contains
  `foreach (k, v; objectAA)` which the -preview=safer checker rejects in
  core.internal.newaa.opApply. std.process.environment is excluded for the
  same reason (opApply over env vars). Both are replaced with minimal
  alternatives using only core.* and std.string.
+/
module hf_download;

import std.stdio        : writefln, writeln, stderr;
import std.string       : endsWith, indexOf, toStringz, strip;
import std.conv         : to;
import std.file         : mkdirRecurse, exists;
import std.path         : buildPath, baseName;
import core.stdc.stdio  : fgets, printf, snprintf;
import core.sys.posix.stdio : popen, pclose;
import core.stdc.stdlib : system;
import core.stdc.string : strlen;

// Use POSIX getenv directly — std.process.environment.opApply iterates env
// vars via an AA, which fails under -preview=safer at the stdlib level.
version(Posix)
    import core.sys.posix.stdlib : c_getenv = getenv;
else version(Windows)
    extern(C) char* getenv(scope const char*) nothrow @nogc;

enum HF_API  = "https://huggingface.co/api";
enum HF_BASE = "https://huggingface.co";

int main(string[] args)
{
    string repo;
    string filename;
    string outDir = ".";
    string token  = envGet("HF_TOKEN");
    bool   listAll = false;

    for (int i = 1; i < cast(int) args.length; i++)
    {
        switch (args[i])
        {
            case "-r":
                if (++i < cast(int) args.length) repo = args[i];
                else return printUsage(args[0]);
                break;
            case "-f":
                if (++i < cast(int) args.length) filename = args[i];
                else return printUsage(args[0]);
                break;
            case "-o":
                if (++i < cast(int) args.length) outDir = args[i];
                else return printUsage(args[0]);
                break;
            case "-t":
                if (++i < cast(int) args.length) token = args[i];
                else return printUsage(args[0]);
                break;
            case "-l", "--list":
                listAll = true;
                break;
            case "-h", "--help":
                return printUsage(args[0]);
            default:
                stderr.writefln("unknown option: %s", args[i]);
                return printUsage(args[0]);
        }
    }

    if (repo.length == 0)
    {
        stderr.writeln("error: -r owner/repo is required");
        return printUsage(args[0]);
    }

    if (filename.length == 0 || listAll)
        return listGgufFiles(repo, token);
    else
        return downloadFile(repo, filename, outDir, token);
}

// ── List .gguf files in a repository ─────────────────────────────────────────

int listGgufFiles(string repo, string token)
{
    immutable url = HF_API ~ "/models/" ~ repo;

    string body_;
    try
        body_ = httpGet(url, token);
    catch (Exception e)
    {
        stderr.writefln("error: request failed: %s", e.msg);
        return 1;
    }

    // Check for API-level error field before extracting names.
    if (hasJsonKey(body_, "error"))
    {
        stderr.writefln("error from HuggingFace API: %s",
                        extractJsonString(body_, "error"));
        return 1;
    }
    if (!hasJsonKey(body_, "siblings"))
    {
        stderr.writeln("error: unexpected API response — no 'siblings' field");
        return 1;
    }

    auto files = extractGgufNames(body_);
    insertionSort(files);

    if (files.length == 0)
    {
        writefln("No .gguf files found in %s", repo);
        return 0;
    }

    writefln("GGUF files in %s  (%d found)", repo, files.length);
    writeln("─────────────────────────────────────────────");
    foreach (f; files)
        writefln("  %s", f);
    writeln();
    writefln("Download: hf-download -r %s -f <filename>", repo);
    return 0;
}

// ── Download a single file ────────────────────────────────────────────────────

int downloadFile(string repo, string filename, string outDir, string token)
{
    immutable url     = HF_BASE ~ "/" ~ repo ~ "/resolve/main/" ~ filename;
    immutable outPath = buildPath(outDir, baseName(filename));

    if (outDir != "." && !exists(outDir))
        mkdirRecurse(outDir);

    writefln("Downloading: %s", filename);
    writefln("  from : %s", url);
    writefln("  to   : %s", outPath);
    writeln();

    // curl --progress-bar writes to stderr directly — live progress in terminal.
    string cmd = "curl -L --progress-bar";
    if (token.length)
        cmd ~= " -H \"Authorization: Bearer " ~ token ~ "\"";
    cmd ~= " -o \"" ~ outPath ~ "\" \"" ~ url ~ "\"";

    // system() inherits stdout/stderr so curl's progress bar is visible.
    int rc = systemCmd(cmd);
    if (rc != 0)
    {
        stderr.writefln("error: curl exited with code %d", rc);
        return 1;
    }

    writefln("\nSaved: %s", outPath);
    return 0;
}

// ── HTTP GET via curl CLI ─────────────────────────────────────────────────────
// Uses popen+curl instead of std.net.curl.HTTP.
// std.net.curl.HTTP iterates its internal header AA with foreach(k,v;aa),
// which the -preview=safer checker rejects in core.internal.newaa.opApply.

string httpGet(string url, string token) @trusted
{
    string cmd = "curl -sf -L";
    if (token.length)
        cmd ~= " -H \"Authorization: Bearer " ~ token ~ "\"";
    cmd ~= " \"" ~ url ~ "\"";

    auto pipe = popen(cmd.toStringz, "r");
    if (pipe is null)
        throw new Exception("popen failed");

    // Plain ~= instead of appender!string — appender instantiates
    // std.typecons.RefCounted which triggers typecons.d(1279) under -preview=all.
    string result;
    char[4096] tmp = void;
    while (fgets(tmp.ptr, cast(int) tmp.sizeof, pipe) !is null)
        result ~= tmp.ptr[0 .. strlen(tmp.ptr)];

    int rc = pclose(pipe);
    if (rc != 0)
        throw new Exception("curl exited with code " ~ rc.to!string);

    return result;
}

int systemCmd(string cmd) @trusted nothrow
{
    return system(cmd.toStringz);
}

// ── Minimal JSON helpers — no std.json ───────────────────────────────────────
// std.json.JSONValue.toString() contains `foreach (k, v; objectAA)` which
// fails under -preview=safer even when never called. Use string scanning only.

/// True if the JSON text contains `"key"` at the top level.
bool hasJsonKey(string json, string key) pure nothrow @safe
{
    return json.indexOf("\"" ~ key ~ "\"") >= 0;
}

/// Extract the string value of the first occurrence of `"key": "value"`.
string extractJsonString(string json, string key) pure @safe
{
    auto needle = "\"" ~ key ~ "\"";
    auto kpos   = json.indexOf(needle);
    if (kpos < 0) return "";
    auto pos = kpos + needle.length;
    while (pos < json.length && (json[pos] == ' ' || json[pos] == ':')) pos++;
    if (pos >= json.length || json[pos] != '"') return "";
    pos++;
    auto end = json.indexOf('"', pos);
    if (end < 0) return "";
    return json[pos .. end];
}

// Simple insertion sort — avoids std.algorithm.sort which fails under
// -preview=safer: binaryFun calls lessFun(r.front, r.front), which the
// aliasing checker rejects as two mutable references to the same value.
void insertionSort(string[] arr) @safe nothrow
{
    for (size_t i = 1; i < arr.length; i++)
    {
        string key = arr[i];
        ptrdiff_t j = cast(ptrdiff_t) i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/// Scan the HF API response for all `"rfilename": "*.gguf"` entries.
string[] extractGgufNames(string json) pure @safe
{
    string[] names;
    enum needle = "\"rfilename\"";
    ptrdiff_t pos = 0;
    while (true)
    {
        auto kpos = json.indexOf(needle, pos);
        if (kpos < 0) break;
        pos = kpos + needle.length;
        // skip whitespace and ':'
        while (pos < json.length &&
               (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t'))
            pos++;
        if (pos >= json.length || json[pos] != '"') continue;
        pos++; // skip opening quote
        auto end = json.indexOf('"', pos);
        if (end < 0) break;
        auto name = json[pos .. end];
        if (name.endsWith(".gguf"))
            names ~= name;
        pos = end + 1;
    }
    return names;
}

// ── Environment variable helper ───────────────────────────────────────────────

string envGet(string name) @trusted nothrow
{
    version(Posix)
        auto p = c_getenv(name.toStringz);
    else
        auto p = getenv(name.toStringz);
    if (p is null) return "";
    return cast(string) p[0 .. strlen(p)];
}

// ── Byte formatting ───────────────────────────────────────────────────────────
// Uses printf/snprintf throughout — writefln with float format specifiers
// fails under -preview=all due to std.format.internal.floats aliasing issues.

// Two alternating static buffers: char[32][2] in D means 2 × char[32].
const(char)* fmtBytes(ulong n) @trusted nothrow
{
    static char[32][2] bufs;
    static int which = 0;
    which = 1 - which;
    char* b = bufs[which].ptr;

    double gb = cast(double) n / (1024.0 * 1024 * 1024);
    double mb = cast(double) n / (1024.0 * 1024);
    double kb = cast(double) n / 1024.0;

    if (n >= 1024UL * 1024 * 1024)
        snprintf(b, 32, "%.2f GiB", gb);
    else if (n >= 1024UL * 1024)
        snprintf(b, 32, "%.2f MiB", mb);
    else if (n >= 1024)
        snprintf(b, 32, "%.2f KiB", kb);
    else
        snprintf(b, 32, "%llu B", cast(ulong) n);

    return b;
}

string humanBytes(ulong n) @trusted nothrow
{
    auto p = fmtBytes(n);
    return cast(string) p[0 .. strlen(p)];
}

// ── Usage ─────────────────────────────────────────────────────────────────────

int printUsage(string prog) @trusted nothrow
{
    // Use toStringz — prog.ptr is not null-terminated; printf would read past it.
    auto p = prog.toStringz;
    printf(
        "\nusage: %s -r owner/repo [-f file] [-o outdir] [-t token]\n\n"
        ~ "  -r   HuggingFace repository  (e.g. unsloth/Qwen3-0.6B-GGUF)\n"
        ~ "  -f   filename to download    (omit to list .gguf files)\n"
        ~ "  -o   output directory        (default: current directory)\n"
        ~ "  -t   HF access token         (or set HF_TOKEN env var)\n"
        ~ "  -l   list .gguf files and exit\n"
        ~ "  -h   show this help\n\n"
        ~ "examples:\n"
        ~ "  %s -r unsloth/Qwen3-0.6B-GGUF\n"
        ~ "  %s -r unsloth/Qwen3-0.6B-GGUF -f Qwen3-0.6B-Q4_K_M.gguf -o ~/models\n"
        ~ "  HF_TOKEN=hf_xxx %s -r org/private-model -f model.gguf\n\n",
        p, p, p, p);
    return 1;
}
