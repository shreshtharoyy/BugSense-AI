CHROMA_PATH = "./data/chromadb"
COLLECTION_NAME = "bug_reports"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE is asymmetric: only queries carry this prefix, never documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Second-stage cross-encoder. The bi-encoder above scores query and document
# independently (fast over 24k vectors, blind to token-level interaction); the
# reranker reads the (query, document) pair jointly over the top RERANK_POOL dense
# candidates. bge-reranker-base fits a 6GB GPU comfortably; swap to bge-reranker-v2-m3
# for higher quality at more VRAM/latency.
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_POOL = 50
# Cross-encoder predict batch. Larger keeps the GPU fed; 64 pairs of <=512 tokens fit a
# 6GB card in fp16 with room to spare.
RERANK_BATCH_SIZE = 64
# Chars kept per side before reranking. The cross-encoder must fit BOTH texts in 512
# tokens; feeding two full bug reports squeezes each to ~half and drops recall. The
# composed doc is "Title: ... Error Log: ... Description: ...", so the first ~700 chars
# keep the discriminative Title+Error Log (and a little Description) while leaving both
# sides well inside the token budget.
RERANK_MAX_CHARS = 700

# Hybrid retrieval: dense (semantic) and BM25 (exact keyword) each return HYBRID_POOL
# candidates, fused by Reciprocal Rank Fusion. BM25 catches the rare exact tokens
# (NS_ERROR_FAILURE, error codes) that dense embeddings blur together.
HYBRID_POOL = 50
# RRF constant. score(id) = sum over lists of weight/(RRF_K + rank). 60 is the standard
# value; larger flattens the contribution of top ranks, smaller sharpens it.
RRF_K = 60
# Weight on the BM25 side of the fusion (dense side is fixed at 1.0). Swept over the 3,460
# duplicate pairs: 0.3 beat pure dense on Recall@1/@10 and MRR with Recall@5 unchanged,
# and higher weights regressed -- BM25 helps only as a light rescue signal for exact-token
# cases, not as an equal partner. 0.3 is the tuned optimum.
RRF_BM25_WEIGHT = 0.3

# Hard limit of BAAI/bge-small-en-v1.5. Text past this is dropped silently.
MAX_SEQ_TOKENS = 512

BATCH_SIZE = 16

TOP_K = 5
SIMILARITY_THRESHOLD = 0.60

# Character guards, not token guards: 2000 chars is roughly 500 tokens, so these
# narrow the input but cannot by themselves keep a document under MAX_SEQ_TOKENS.
MAX_DESCRIPTION_LENGTH = 2000
MAX_ERROR_LOG_LENGTH = 1000
