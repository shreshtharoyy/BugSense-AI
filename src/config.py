CHROMA_PATH = "./data/chromadb"
COLLECTION_NAME = "bug_reports"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE is asymmetric: only queries carry this prefix, never documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Hard limit of BAAI/bge-small-en-v1.5. Text past this is dropped silently.
MAX_SEQ_TOKENS = 512

BATCH_SIZE = 16

TOP_K = 5
SIMILARITY_THRESHOLD = 0.60

# Character guards, not token guards: 2000 chars is roughly 500 tokens, so these
# narrow the input but cannot by themselves keep a document under MAX_SEQ_TOKENS.
MAX_DESCRIPTION_LENGTH = 2000
MAX_ERROR_LOG_LENGTH = 1000
