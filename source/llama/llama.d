module llama.llama;

public import c.llama_stubs;

// llama.h #define constants not captured by importC as enums
enum LLAMA_DEFAULT_SEED = 0xFFFF_FFFFu;
enum LLAMA_TOKEN_NULL   = -1;

enum LLAMA_FILE_MAGIC_GGLA = 0x67676c61u; /// 'ggla'
enum LLAMA_FILE_MAGIC_GGSN = 0x6767736eu; /// 'ggsn'
enum LLAMA_FILE_MAGIC_GGSQ = 0x67677371u; /// 'ggsq'

enum LLAMA_SESSION_MAGIC   = LLAMA_FILE_MAGIC_GGSN;
enum LLAMA_SESSION_VERSION = 9;

enum LLAMA_STATE_SEQ_MAGIC   = LLAMA_FILE_MAGIC_GGSQ;
enum LLAMA_STATE_SEQ_VERSION = 2;
