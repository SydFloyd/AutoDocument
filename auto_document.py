"""Module for generating and updating comprehensive documentation of a Python codebase using language models.

This module provides functionality to:
- Cache and retrieve summaries for files and directories to avoid redundant computations.
- Compute SHA-256 hashes of files to detect changes and update summaries accordingly.
- Parse Python files with Abstract Syntax Tree (AST) to extract structured information such as classes, functions, and imports.
- Use a language model (LLM) to summarize large text data by chunking, if necessary, ensuring the summaries are cohesive and informative.
- Generate documentation for individual files and directories, as well as the entire codebase, using a combination of AST parsing and full-text summarization.

Key functions and classes:
- `load_cache` and `save_cache`: Manage local JSON cache for summaries.
- `compute_sha256` and `combine_hashes`: Handle file and directory hash computations.
- `count_tokens` and `chunk_text`: Approximate token counts and split text into manageable chunks.
- `ast_parse_file`: Extract structured information from Python files using AST.
- `summarize_large_text` and `summarize_file_ast`: Create summaries for text and AST-parsed file data.
- `get_file_summary` and `summarize_directory`: Generate or retrieve summaries for files and directories.
- `build_documentation` and `update_documentation`: Orchestrate the documentation process for the entire codebase.

Dependencies:
- `os`, `ast`, `hashlib`, `json`, `collections`: Standard libraries for file operations, hashing, JSON handling, and data structures.
- `tiktoken`: External library for token encoding.
- `src.config` and `src.utils.openai_utils`: Import configuration settings and language model utility functions."""

import os
import ast
import hashlib
import json
from collections import defaultdict
import tiktoken
from llm import LLM

if not os.path.exists("cache"):
    os.mkdir("cache")
CACHE_FILE = "cache/.summary_cache.json"

EXCLUDE_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache"}

CHUNK_SIZE = 32_000 # tokens

def load_cache(cache_file=CACHE_FILE):
    """Load summaries for files and directories from a local JSON cache."""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"files": {}, "directories": {}, "codebase": {}}

def save_cache(cache, cache_file=CACHE_FILE):
    """Save updated cache to disk as JSON."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

def combine_hashes(hashes):
    """Combine multiple hashes into one by hashing the concatenation of them."""
    combined = hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
    return combined

def count_tokens(text: str) -> int:
    """Approximate token count for a given text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    """
    Split text into chunks of up to `chunk_size` tokens.
    Useful if a single summary prompt grows beyond the model context.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    for i in range(0, len(tokens), chunk_size):
        yield enc.decode(tokens[i:i+chunk_size])

def ast_parse_file(filepath: str):
    """
    Parse a Python file using AST to extract structured info.
    Returns a dict with classes, functions, imports, docstring, etc.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": f"Could not parse {filepath}", "source": source}

    classes = []
    functions = []
    imports = []
    docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}")

    return {
        "filepath": filepath,
        "docstring": docstring,
        "classes": classes,
        "functions": functions,
        "imports": list(set(imports)),
        "source": source
    }

def summarize_large_text(llm, text: str, chunk_label: str = "text"):
    """
    Summarize large text by chunking if necessary, returning a cohesive final summary.
    """
    if count_tokens(text) <= CHUNK_SIZE:
        return llm.prompt(f"Summarize the following {chunk_label}:\n{text}")
    else:
        partial_summaries = []
        for idx, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
            summary_part = llm.prompt(
                f"You are summarizing chunk #{idx} of a large {chunk_label}.\n"
                "Focus on key functionalities, classes, dependencies, purpose, etc.\n"
                f"{chunk}"
            )
            partial_summaries.append(summary_part)

        combined_summary = llm.prompt(
            "Combine these chunk-level summaries into one cohesive final summary:\n\n"
            + "\n\n".join(partial_summaries)
        )
        return combined_summary

def summarize_file_ast(llm, file_info: dict):
    """
    Summarize a file using AST data, focusing on classes, functions, imports, etc.
    """
    prompt = (
        "You are generating documentation for this Python file.\n"
        "Summarize key classes, functions, imports, and the file's overall purpose.\n"
        f"File: {file_info['filepath']}\n"
        f"Docstring: {file_info.get('docstring','No top-level docstring.')}\n"
        f"Classes: {file_info.get('classes',[])}\n"
        f"Functions: {file_info.get('functions',[])}\n"
        f"Imports: {file_info.get('imports',[])}\n"
    )
    return llm.prompt(prompt)

def get_file_summary(llm, filepath: str, cache: dict, use_ast=True) -> str:
    """
    Return a cached or newly generated summary for a single file.
    """
    file_hash = compute_sha256(filepath)
    cached_entry = cache["files"].get(filepath)

    # Reuse cached summary if file hash is unchanged
    if cached_entry and cached_entry["hash"] == file_hash:
        print(f"No changes detected in {filepath}; using cached summary.")
        return cached_entry["summary"]
    print(f"Summarizing {filepath}...")

    # Otherwise, summarize anew
    if use_ast:
        file_info = ast_parse_file(filepath)
        if "error" in file_info or not file_info.get("source"):
            # Fallback to full-text summarization
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
            summary = summarize_large_text(llm, file_content, chunk_label="file content")
        else:
            summary = summarize_file_ast(llm, file_info)
    else:
        # Summarize by reading raw content
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        summary = summarize_large_text(llm, file_content, chunk_label="file content")

    # Update cache
    cache["files"][filepath] = {
        "hash": file_hash,
        "summary": summary
    }
    return summary

def summarize_directory(llm, dir_path: str, file_summaries: dict, cache: dict) -> str:
    """
    Summarize a directory based on its file-level summaries (no raw code embedding).
    
    'file_summaries' is a dict: { filepath -> summary_text }.
    """
    # Compute directory hash by combining file hashes
    file_hashes = [cache["files"][fp]["hash"] for fp in file_summaries]
    dir_hash = combine_hashes(sorted(file_hashes))

    # Reuse cached directory summary if hash is unchanged
    cached_entry = cache["directories"].get(dir_path)
    if cached_entry and cached_entry["dir_hash"] == dir_hash:
        print(f"No changes detected in {dir_path}; using cached summary.")
        return cached_entry["summary"]
    print(f"Summarizing {dir_path}...")

    # Build a single text that includes each file-level summary
    prompt_parts = []
    for fp, summary in file_summaries.items():
        prompt_parts.append(f"FILE: {fp}\nSUMMARY:\n{summary}\n")
    combined_summaries = "\n".join(prompt_parts)

    directory_summary = summarize_large_text(llm, combined_summaries, chunk_label="directory summaries")

    # Cache the new summary
    cache["directories"][dir_path] = {
        "dir_hash": dir_hash,
        "summary": directory_summary
    }
    return directory_summary

def build_documentation(root_dir: str, use_ast: bool = True) -> str:
    """
    Main pipeline:
      1) Summarize each file (cached).
      2) Summarize each directory from file-level summaries (cached).
      3) Summarize the entire codebase from directory-level summaries (cached).
    """
    llm = LLM(system_message="You are an expert in generating complete and concise documentation of code.")
    cache = load_cache()

    # Gather Python files
    dir_to_files = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        py_files = [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
        if py_files:
            dir_to_files[dirpath].extend(py_files)

    # Summarize files and directories
    directory_summaries = {}
    for dir_path, files in dir_to_files.items():
        file_sums = {}
        for fpath in files:
            file_sums[fpath] = get_file_summary(llm, fpath, cache, use_ast=use_ast)

        directory_summary = summarize_directory(llm, dir_path, file_sums, cache)
        directory_summaries[dir_path] = directory_summary

    # Summarize the entire codebase
    # Create a codebase hash from directory hashes
    dir_hashes = [cache["directories"][dpath]["dir_hash"] for dpath in directory_summaries]
    codebase_hash = combine_hashes(sorted(dir_hashes))

    cached_codebase = cache["codebase"].get("hash")
    if cached_codebase and cached_codebase == codebase_hash:
        final_summary = cache["codebase"]["summary"]
    else:
        # Build a text from the directory-level summaries
        all_dir_summaries_text = []
        for dpath, dsummary in directory_summaries.items():
            all_dir_summaries_text.append(f"DIR: {dpath}\nSUMMARY:\n{dsummary}\n")
        combined_text = "\n".join(all_dir_summaries_text)

        # Summarize everything in one go (chunked if needed)
        final_summary = summarize_large_text(llm, combined_text, chunk_label="codebase")

        # Cache the final codebase summary
        cache["codebase"]["hash"] = codebase_hash
        cache["codebase"]["summary"] = final_summary

    # Save updated cache
    save_cache(cache)

    return final_summary

def update_documentation(root_dir: str, doc_output_path: str = "docs.md", use_ast: bool = True):
    """
    Regenerates the codebase documentation and saves it to 'docs.md' in the project root.
    
    :param root_dir: Path to the root directory of the project.
    :param use_ast: If True, uses AST-based file summaries where possible.
    """
    print("Updating documentation...")
    final_doc = build_documentation(root_dir, use_ast=use_ast)
    docs_path = os.path.join(root_dir, doc_output_path)
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(final_doc)
    print(f"Documentation saved to {docs_path}.")

    return docs_path
