/++
Download GGUF files from HuggingFace Hub.

Without -f, lists .gguf files with sizes.  With -f, downloads via curl.
Auth: -t token or HF_TOKEN env var (private repos / higher rate limits).

Usage:
  hf-download -r owner/repo [-f file] [-o outdir] [-t token]

Requires curl in PATH. Uses popen/system instead of std.net.curl,
which breaks under -preview=all.
+/
module hf_download;

import std.stdio        : writefln, writeln, stderr;
import std.string       : endsWith, indexOf, toStringz, strip;
import std.conv         : to;
import std.file         : mkdirRecurse, exists;
import std.path         : buildPath, baseName;
import core.stdc.stdio  : fgets, printf, snprintf;
import core.stdc.stdlib : system;
import core.stdc.string : strlen;

// Windows spells these _popen/_pclose.
version(Posix)
{
    import core.sys.posix.stdio : popen, pclose;
}
else version(Windows)
{
    import core.stdc.stdio : FILE;
    extern(C) nothrow @nogc
    {
        FILE* _popen(scope const char* cmd, scope const char* mode);
        int   _pclose(FILE* stream);
    }
    alias popen  = _popen;
    alias pclose = _pclose;
}

// std.process.environment breaks under -preview=all.
version(Posix)
    import core.sys.posix.stdlib : c_getenv = getenv;
else version(Windows)
    extern(C) char* getenv(scope const char*) nothrow @nogc;

enum HF_API  = "https://huggingface.co/api";
enum HF_BASE = "https://huggingface.co";

struct GgufFile
{
    string name;
    ulong  size; // bytes; 0 if unavailable
}

int main(string[] args)
{
    string repo;
    string filename;
    string outDir  = ".";
    string token   = envGet("HF_TOKEN");
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

// ── List ──────────────────────────────────────────────────────────────────────

int listGgufFiles(string repo, string token)
{
    // ?blobs=true includes LFS size for each sibling.
    immutable url = HF_API ~ "/models/" ~ repo ~ "?blobs=true";

    string body_;
    try
        body_ = httpGet(url, token);
    catch (Exception e)
    {
        stderr.writefln("error: %s", e.msg);
        return 1;
    }

    if (hasJsonKey(body_, "error"))
    {
        stderr.writefln("API error: %s", extractJsonString(body_, "error"));
        return 1;
    }
    if (!hasJsonKey(body_, "siblings"))
    {
        stderr.writeln("error: unexpected API response");
        return 1;
    }

    auto files = extractGgufFiles(body_);
    insertionSort(files);

    if (files.length == 0)
    {
        writefln("No .gguf files found in %s", repo);
        return 0;
    }

    int maxLen = 0;
    foreach (ref f; files)
        if (cast(int) f.name.length > maxLen)
            maxLen = cast(int) f.name.length;

    writefln("GGUF files in %s  (%d found)", repo, files.length);
    printFileList(files, maxLen);
    writeln();
    writefln("Download: hf-download -r %s -f <filename>", repo);
    return 0;
}

// ── Download ──────────────────────────────────────────────────────────────────

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

    // --progress-bar writes to stderr so it shows live in the terminal.
    string cmd = "curl -L --progress-bar";
    if (token.length)
        cmd ~= " -H \"Authorization: Bearer " ~ token ~ "\"";
    cmd ~= " -o \"" ~ outPath ~ "\" \"" ~ url ~ "\"";

    int rc = systemCmd(cmd);
    if (rc != 0)
    {
        stderr.writefln("error: curl exited with code %d", rc);
        return 1;
    }

    writefln("\nSaved: %s", outPath);
    return 0;
}

// ── HTTP ──────────────────────────────────────────────────────────────────────

// popen+curl GET; std.net.curl breaks under -preview=all.
string httpGet(string url, string token) @trusted
{
    string cmd = "curl -sf -L";
    if (token.length)
        cmd ~= " -H \"Authorization: Bearer " ~ token ~ "\"";
    cmd ~= " \"" ~ url ~ "\"";

    auto pipe = popen(cmd.toStringz, "r");
    if (pipe is null)
        throw new Exception("popen failed");

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

// ── JSON helpers ──────────────────────────────────────────────────────────────
// No std.json — breaks under -preview=all.

bool hasJsonKey(string json, string key) pure nothrow @safe
{
    return json.indexOf("\"" ~ key ~ "\"") >= 0;
}

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

// Parses name + byte size from each sibling in a ?blobs=true API response.
GgufFile[] extractGgufFiles(string json) pure @safe
{
    GgufFile[] files;
    enum rfnKey = "\"rfilename\"";
    enum szKey  = "\"size\"";
    ptrdiff_t pos = 0;
    while (true)
    {
        auto kpos = json.indexOf(rfnKey, pos);
        if (kpos < 0) break;
        pos = kpos + rfnKey.length;

        while (pos < json.length &&
               (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t'))
            pos++;
        if (pos >= json.length || json[pos] != '"') continue;
        pos++;
        auto end = json.indexOf('"', pos);
        if (end < 0) break;
        auto name = json[pos .. end];
        pos = end + 1;

        if (!name.endsWith(".gguf")) continue;

        // Search for "size" only within this sibling's object, not the next.
        auto nextRfn  = json.indexOf(rfnKey, pos);
        auto searchTo = nextRfn < 0 ? cast(ptrdiff_t) json.length : nextRfn;

        ulong sz  = 0;
        auto  szp = json.indexOf(szKey, pos);
        if (szp >= 0 && szp < searchTo)
        {
            auto p = szp + szKey.length;
            while (p < json.length &&
                   (json[p] == ' ' || json[p] == ':' || json[p] == '\t'))
                p++;
            ulong n = 0;
            bool  ok = false;
            while (p < json.length && json[p] >= '0' && json[p] <= '9')
            {
                n  = n * 10 + (json[p] - '0');
                p++;
                ok = true;
            }
            if (ok) sz = n;
        }

        files ~= GgufFile(name, sz);
    }
    return files;
}

// std.algorithm.sort breaks under -preview=all.
void insertionSort(GgufFile[] arr) @safe nothrow
{
    for (size_t i = 1; i < arr.length; i++)
    {
        GgufFile key = arr[i];
        ptrdiff_t j  = cast(ptrdiff_t) i - 1;
        while (j >= 0 && arr[j].name > key.name)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void printFileList(GgufFile[] files, int maxLen) @trusted nothrow
{
    enum sep = "─";
    for (int i = 0; i < maxLen + 12; i++) printf("%s", sep.ptr);
    printf("\n");

    ulong total = 0;
    foreach (ref f; files)
    {
        total += f.size;
        if (f.size > 0)
            printf("  %-*s  %s\n", maxLen, f.name.toStringz, fmtBytes(f.size));
        else
            printf("  %s\n", f.name.toStringz);
    }

    if (total > 0)
        printf("  %*s  %s\n", maxLen, "Total".ptr, fmtBytes(total));
}

// ── Env ───────────────────────────────────────────────────────────────────────

string envGet(string name) @trusted nothrow
{
    version(Posix)
        auto p = c_getenv(name.toStringz);
    else
        auto p = getenv(name.toStringz);
    if (p is null) return "";
    return cast(string) p[0 .. strlen(p)];
}

// ── Formatting ────────────────────────────────────────────────────────────────
// printf only; writefln+float breaks under -preview=all.

// Two alternating buffers so a single printf can call fmtBytes() twice.
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
    // prog.ptr is not null-terminated; toStringz makes a safe C string.
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
