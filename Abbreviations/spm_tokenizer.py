"""
SentencePiece tokenizer wrapper for pinyin-initials encoder-decoder model.

Provides a clean interface to SentencePiece tokenization with special token handling.
"""
from typing import List
import sentencepiece as spm


class SentencePieceTokenizer:
    """
    Wrapper around SentencePiece for encoding/decoding text.
    
    Handles special tokens (PAD, UNK, BOS, EOS) and provides convenient
    methods for tokenization in encoder-decoder models.
    """
    
    def __init__(self, model_path: str):
        """
        Load SentencePiece model.
        
        Args:
            model_path: Path to .model file (e.g., "outputs/spm_model.model")
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        
        # Get special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.GetPieceSize()
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: If True, prepend BOS token
            add_eos: If True, append EOS token
        
        Returns:
            List of token IDs
        
        Example:
            >>> tokenizer = SentencePieceTokenizer("spm_model.model")
            >>> ids = tokenizer.encode("zh sh r", add_bos=True, add_eos=True)
            >>> # Returns: [BOS_ID, ...token_ids..., EOS_ID]
        """
        ids = self.sp.EncodeAsIds(text)
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        
        Example:
            >>> text = tokenizer.decode([10, 20, 30])
            >>> # Returns: "zh sh r" (or similar)
        """
        return self.sp.DecodeIds(ids)
    
    def id_to_piece(self, token_id: int) -> str:
        """
        Convert single token ID to piece string.
        
        Args:
            token_id: Token ID
        
        Returns:
            Token piece string
        
        Example:
            >>> piece = tokenizer.id_to_piece(10)
            >>> # Returns: "zh" (or similar subword)
        """
        return self.sp.IdToPiece(token_id)
    
    def piece_to_id(self, piece: str) -> int:
        """
        Convert piece string to token ID.
        
        Args:
            piece: Token piece string
        
        Returns:
            Token ID
        """
        return self.sp.PieceToId(piece)
    
    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input text strings
            add_bos: If True, prepend BOS to each sequence
            add_eos: If True, append EOS to each sequence
        
        Returns:
            List of token ID lists
        
        Example:
            >>> texts = ["zh sh r", "b p m f"]
            >>> batch_ids = tokenizer.encode_batch(texts, add_bos=True, add_eos=True)
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
