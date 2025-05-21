import json
import re
from tqdm import tqdm

stock_map = {
    "VCB": ["Vietcombank", "Ngân hàng Vietcombank", "VCB"],
    "CTG": ["VietinBank", "Ngân hàng Công Thương", "Ngân hàng VietinBank", "CTG"],
    "VPB": ["VPBank", "Ngân hàng VPBank", "VPB"],
    "BID": ["BIDV", "Ngân hàng BIDV", "BID"],
    "TCB": ["Techcombank", "Ngân hàng Techcombank", "TCB"],
    "HPG": ["Hòa Phát", "Tập đoàn Hòa Phát", "HPG"],
    "MWG": ["Thế Giới Di Động", "Công ty Thế Giới Di Động", "MWG"],
    "VNM": ["Vinamilk", "Công ty sữa Vinamilk", "VNM"],
    "VIC": ["Vingroup", "Tập đoàn Vingroup", "VIC"],
    "FPT": ["FPT", "Công ty FPT", "FPT Corporation"],
}
# VPB, BID, TCB
def extract_sentences(text):
    """Split text into sentences with improved handling for Vietnamese text."""
    # Clean the text first
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Common Vietnamese abbreviations to handle
    abbreviations = [r'TP\.', r'Tp\.', r'TS\.', r'PGS\.', r'ThS\.', r'ĐH\.', 
                     r'ĐHQG\.', r'TT\.', r'P\.', r'Q\.', r'CT\.', r'CTCP\.', r'UBND\.']
    
    # Replace periods in abbreviations temporarily to avoid splitting sentences there
    for abbr in abbreviations:
        text = re.sub(abbr, abbr.replace('.', '<period>'), text)
    
    # Split text into sentences
    # Look for end punctuation followed by space and capital letter or digit
    pattern = r'([.!?])\s+([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9])'
    text = re.sub(pattern, r'\1\n\2', text)
    
    # Split by newline which now represents sentence boundaries
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    # Restore periods in abbreviations
    result = []
    for sentence in sentences:
        sentence = sentence.replace('<period>', '.')
        result.append(sentence)
    
    return result

def extract_stock_chunks(article_text, stock_map, context_window=2):
    """Find stock codes in an article and return continuous text chunks containing the mentions and their context.
    
    Args:
        article_text: The text of the article to analyze
        stock_map: Dictionary mapping stock codes to their aliases
        context_window: Number of sentences before and after to include as context
        
    Returns:
        Dictionary with stock codes as keys and lists of text chunks as values
    """
    results = {}
    sentences = extract_sentences(article_text)
    
    # For each stock code, find mentions and extract surrounding chunks
    for stock_code, aliases in stock_map.items():
        stock_mention_indices = set()
        
        # Find sentences with direct mentions
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for alias in aliases:
                pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                if re.search(pattern, sentence_lower):
                    stock_mention_indices.add(i)
                    break
        
        # If we found mentions, extract text chunks
        if stock_mention_indices:
            # Group adjacent mentions together to avoid overlapping chunks
            mention_groups = []
            current_group = []
            
            for idx in sorted(stock_mention_indices):
                if not current_group or idx <= current_group[-1] + context_window + 1:
                    # This mention is close to the previous one, add to current group
                    current_group.append(idx)
                else:
                    # This mention is far from previous ones, start new group
                    mention_groups.append(current_group)
                    current_group = [idx]
            
            if current_group:  # Add the last group
                mention_groups.append(current_group)
            
            # Extract text chunks based on mention groups
            stock_chunks = []
            for group in mention_groups:
                # Determine the chunk boundaries with context
                start_idx = max(0, min(group) - context_window)
                end_idx = min(len(sentences), max(group) + context_window + 1)
                
                # Join the sentences to form a continuous chunk
                chunk = " ".join(sentences[start_idx:end_idx])
                stock_chunks.append(chunk)
            
            if stock_chunks:
                results[stock_code] = stock_chunks
    
    return results

print(extract_stock_chunks("Mã FPT tăng vượt trội. nhưng có thể giảm trong nhiều ngày nữa", stock_map))