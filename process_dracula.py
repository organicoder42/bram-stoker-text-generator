#!/usr/bin/env python3
"""
Process Dracula text for model training:
1. Remove Project Gutenberg headers and footers
2. Clean and normalize the text
3. Chunk into consistent passages
"""

import re
import os

def clean_dracula_text(input_file, output_file):
    """Clean the raw Dracula text by removing headers and footers."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find the start of the actual book content
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        start_idx = text.find('\n', start_idx) + 1
    else:
        start_idx = 0
        
    if end_idx != -1:
        text = text[start_idx:end_idx]
    else:
        text = text[start_idx:]
    
    # Clean up the text
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = text.strip()
    
    # Remove chapter headings that are just decorative
    text = re.sub(r'\n\s*\[Illustration:[^\]]*\]\s*\n', '\n\n', text)
    
    # Save cleaned text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Cleaned text saved to {output_file}")
    print(f"Text length: {len(text):,} characters")
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Chunk text into passages of consistent length with overlap."""
    chunks = []
    words = text.split()
    
    i = 0
    while i < len(words):
        # Get chunk_size words
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        
        # Only add if chunk is substantial
        if len(chunk_words) > chunk_size // 2:
            chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap) to create overlap
        i += chunk_size - overlap
        
        if i + chunk_size > len(words):
            break
    
    return chunks

def save_chunks(chunks, output_file):
    """Save chunks to a file, one per line."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            # Escape newlines within chunks
            chunk_escaped = chunk.replace('\n', '\\n')
            f.write(f"{chunk_escaped}\n")
    
    print(f"Saved {len(chunks)} chunks to {output_file}")
    print(f"Average chunk length: {sum(len(chunk) for chunk in chunks) // len(chunks):,} characters")

def main():
    input_file = "dracula_raw.txt"
    cleaned_file = "dracula_cleaned.txt"
    chunks_file = "dracula_chunks.txt"
    
    # Step 1: Clean the text
    print("Step 1: Cleaning text...")
    cleaned_text = clean_dracula_text(input_file, cleaned_file)
    
    # Step 2: Chunk the text
    print("\nStep 2: Chunking text...")
    chunks = chunk_text(cleaned_text, chunk_size=1000, overlap=100)
    
    # Step 3: Save chunks
    print("\nStep 3: Saving chunks...")
    save_chunks(chunks, chunks_file)
    
    print(f"\nProcessing complete!")
    print(f"Files created:")
    print(f"- {cleaned_file}: Cleaned full text")
    print(f"- {chunks_file}: Text chunks for training")

if __name__ == "__main__":
    main()