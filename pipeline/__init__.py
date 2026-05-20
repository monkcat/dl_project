"""End-to-end RAG pipeline modules.

Stage A: extraction, graph_builder
Stage B: encoding (indexed: content + element-type + section-role)
Stage C: retrieval (coarse top-K, content-only)
Stage D: rerank (anchor-relative score using HS-PE + graph), expansion (PQ pack)
Stage E: generation (VLM)
"""
