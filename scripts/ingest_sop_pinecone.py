import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import pypdf  

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Suppress the Windows symlink warning noise completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# ==========================================
# 1. PATH RESOLUTION & SETUP
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent 
load_dotenv(project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Vector store backend: PINECONE (cloud, default) or FAISS (local files, no account needed)
VECTOR_DB = os.getenv("VECTOR_DB", "PINECONE").strip().upper()

if VECTOR_DB == "PINECONE" and not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env (or set VECTOR_DB=FAISS to use a local index)")

# Set up clean production data cache path (separate cache per backend so switching never skips files)
cache_dir = project_root / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
HASH_CACHE_FILE = cache_dir / ("faiss_hash_cache.json" if VECTOR_DB == "FAISS" else "ingestion_hash_cache.json")

hash_cache = {}
if HASH_CACHE_FILE.exists():
    try:
        with open(HASH_CACHE_FILE, "r") as f:
            hash_cache = json.load(f)
    except Exception:
        hash_cache = {}

# ==========================================
# 2. DYNAMIC ENVIRONMENT ROUTING
# ==========================================
EMBEDDINGS_MODEL_SETTING = os.getenv("Embeddings_model", "LOCAL").strip().upper()

if EMBEDDINGS_MODEL_SETTING == "OPENAI":
    print("🤖 Mode: Utilizing Cloud OpenAI Embeddings (1536 Dim)...")
    embeddings = OpenAIEmbeddings()
    INDEX_NAME = "fde-sop-index-openai"  # Isolated OpenAI Index
    TARGET_DIMENSION = 1536
else:
    # Read the explicit model identifier casing string from the .env parameters
    local_model_target = os.getenv("Local_Embedding_Model", "BAAI/bge-m3").strip()
    
    print(f"🤗 Mode: Local Fallback Settings Activated. Launching [{local_model_target}] (1024 Dim)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_target,   # Passes parameter dynamically
        model_kwargs={'device': 'cpu'}
    )
    INDEX_NAME = "fde-sop-index-local"   # Isolated Local Model Index
    TARGET_DIMENSION = 1024              # Standard width for BGE-M3

# ==========================================
# 3. PINECONE PROVISIONING
# ==========================================
FAISS_DIR = project_root / "data" / "vector_store" / INDEX_NAME

if VECTOR_DB == "FAISS":
    print(f"📁 Local FAISS index target: [{FAISS_DIR}]")
else:
    print(f"Connecting to Pinecone Index Target: [{INDEX_NAME}]...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    # Self-healing verification in case an index was created with a wrong legacy dimension
    if INDEX_NAME in existing_indexes:
        desc = pc.describe_index(INDEX_NAME)
        if desc.dimension != TARGET_DIMENSION:
            print(f"⚠️ Fixing tracking: Purging mismatched {desc.dimension} dim index...")
            pc.delete_index(INDEX_NAME)
            existing_indexes = [name for name in existing_indexes if name != INDEX_NAME]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating isolated target index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=TARGET_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index_client = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# ==========================================
# 4. ROBUST POLYMORPHIC PARSER
# ==========================================
def parse_and_chunk_document(doc_path: Path) -> list[Document]:
    ext = doc_path.suffix.lower()
    raw_chunks: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    
    if ext == ".md":
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_text = doc_path.read_text(encoding="utf-8")
        header_docs = md_splitter.split_text(raw_text)
        raw_chunks = text_splitter.split_documents(header_docs)
        
    elif ext == ".txt":
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_docs = [Document(page_content=raw_text)]
        raw_chunks = text_splitter.split_documents(raw_docs)
        
    elif ext == ".pdf":
        pdf_docs = []
        try:
            with open(doc_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pdf_docs.append(Document(page_content=page_text, metadata={"page_number": page_num + 1}))
            raw_chunks = text_splitter.split_documents(pdf_docs)
        except Exception as e:
            print(f"  ❌ Error parsing PDF {doc_path.name}: {e}")
            return []
        
    elif ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(doc_path) if ext == ".csv" else pd.read_excel(doc_path)
        except Exception as e:
            print(f"  ❌ Error reading table: {e}")
            return []
            
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_items = [
                f"{str(col)}: {str(val)}" 
                for col, val in row_dict.items() 
                if pd.notna(val) and str(val).strip() != ""
            ]
            
            if row_items:
                row_text = " | ".join(row_items)
                doc_item = Document(page_content=row_text, metadata={"row_index": int(idx)})
                raw_chunks.append(doc_item)

    # sanity check
    valid_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()
        if clean_text:
            chunk.page_content = clean_text
            valid_chunks.append(chunk)
            
    return valid_chunks

# ==========================================
# 5. INCREMENTAL PIPELINE WITH BATCHING
# ==========================================
policy_dir = project_root / "data" / "policy"
target_patterns = ["*.md", "*.txt", "*.pdf", "*.csv", "*.xlsx"]
current_files = {}
for pattern in target_patterns:
    for file_path in policy_dir.glob(pattern):
        current_files[file_path.name] = file_path

print(f"Found {len(current_files)} policy file(s) in {policy_dir}...")

# ------------------------------------------
# 5a. LOCAL FAISS PATH
# FAISS has no metadata-filtered delete, so when any policy file is added, changed
# or removed we rebuild the whole local index (fine for a handful of SOP files).
# ------------------------------------------
if VECTOR_DB == "FAISS":
    from langchain_community.vectorstores import FAISS

    current_hashes = {name: hashlib.md5(path.read_bytes()).hexdigest() for name, path in current_files.items()}
    if current_hashes == hash_cache and (FAISS_DIR / "index.faiss").exists():
        print("🌴 Local FAISS index is already up-to-date.")
        raise SystemExit(0)

    all_chunks, all_ids = [], []
    for file_name, file_path in current_files.items():
        print(f"🔄 Processing: {file_name}...")
        chunks = parse_and_chunk_document(file_path)
        if not chunks:
            print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
            current_hashes.pop(file_name, None)
            continue
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = file_path.suffix.replace(".", "").upper()
            chunk.metadata["document_type"] = "Compliance Asset"
            all_ids.append(f"{file_name}-chunk-{idx}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No text chunks found in {policy_dir}; nothing to index.")

    print(f"  📤 Embedding {len(all_chunks)} chunk(s) into local FAISS index...")
    faiss_store = FAISS.from_documents(all_chunks, embeddings, ids=all_ids)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(FAISS_DIR))

    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(current_hashes, f, indent=4)
    print(f"✅ FAISS index saved to {FAISS_DIR}")
    raise SystemExit(0)

# ------------------------------------------
# 5b. PINECONE PATH (incremental upserts)
# ------------------------------------------
updated_cache = {}
cache_modified = False

cached_filenames = set(hash_cache.keys())
current_filenames = set(current_files.keys())
deleted_files = cached_filenames - current_filenames

for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        index_client.delete(filter={"source_file": {"$eq": deleted_file}})
        cache_modified = True
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")

for file_name, file_path in current_files.items():
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    updated_cache[file_name] = file_hash
    
    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        continue
        
    print(f"🔄 Processing updates: {file_name}...")
    cache_modified = True
    
    try:
        try:
            index_client.delete(filter={"source_file": {"$eq": file_name}})
        except Exception:
            pass 
        
        chunks = parse_and_chunk_document(file_path)
        if not chunks:
            print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
            continue
            
        explicit_ids = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = file_path.suffix.replace(".", "").upper()
            chunk.metadata["document_type"] = "Compliance Asset"
            explicit_ids.append(f"{file_name}-chunk-{idx}")
            
        batch_size = 100
        total_chunks = len(chunks)
        print(f"  📤 Upserting {total_chunks} chunk(s) in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch_docs = chunks[i : i + batch_size]
            batch_ids = explicit_ids[i : i + batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            
    except Exception as e:
        print(f"❌ Error during ingestion of {file_name}: {e}")
        updated_cache.pop(file_name, None)

# ==========================================
# 6. SYNC HASH CACHE
# ==========================================
if cache_modified:
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(updated_cache, f, indent=4)
    print("✅ Ingestion & cache update complete.")
else:
    print("🌴 Index is already up-to-date.")