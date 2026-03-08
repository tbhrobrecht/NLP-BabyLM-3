# evaluate_blimp.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import os
import glob
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, DebertaV2Tokenizer
from functools import partial
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--subsets', type=str, default="", help="Comma-separated list of subset names (without .jsonl extension). If empty, auto-discover all .jsonl files.")
parser.add_argument('--spm_path', type=str, default="", help="Path to SentencePiece model file. If empty, will check model_path for spm.model or use AutoTokenizer.")
parser.add_argument('--device', type=str, default="", help="Device to run on (cuda/cpu). If empty, auto-detect.")


# ── Custom BabyLM checkpoint support ─────────────────────────────────────────

class _SPMTokenizerWrapper:
    """Thin SentencePiece wrapper with a HF-tokenizer-compatible interface."""
    def __init__(self, spm_path, bos_id=2, eos_id=3, pad_id=0):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(spm_path)
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id
        self.mask_token_id = None
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'

    def encode(self, text, add_special_tokens=True):
        tokens = self._sp.Encode(text, out_type=int)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def __len__(self):
        return self._sp.GetPieceSize()


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.register_buffer('pe', torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _BabyLMEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_seq_len, pad_idx=0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = _PositionalEncoding(d_model, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, src, src_key_padding_mask=None):
        x = self.pos_encoding(self.token_embedding(src))
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


class _ProbabilisticEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.posterior_proj = nn.Linear(d_model, vocab_size)
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, enc_out):
        probs = F.softmax(self.posterior_proj(enc_out), dim=-1)
        soft_emb = probs @ self.vocab_embedding.weight
        return self.fusion_norm(enc_out + soft_emb)


class _BabyLMDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_seq_len, pad_idx=0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = _PositionalEncoding(d_model, max_seq_len)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model))
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        x = self.pos_encoding(self.token_embedding(tgt))
        dec_out = self.transformer_decoder(
            x, memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        mem_ctx = memory.mean(dim=1, keepdim=True).expand_as(dec_out)
        return self.output_proj(self.gate(torch.cat([dec_out, mem_ctx], dim=-1)))


class _BabyLMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v, d, h = cfg['vocab_size'], cfg['d_model'], cfg['n_heads']
        ff, drop = cfg['d_ff'], cfg['dropout']
        ne, nd, ml = cfg['n_encoder_layers'], cfg['n_decoder_layers'], cfg['max_seq_len']
        self.encoder = _BabyLMEncoder(v, d, h, ff, ne, drop, ml, cfg['pad_id'])
        self.prob_encoder = _ProbabilisticEncoder(v, d)
        self.decoder = _BabyLMDecoder(v, d, h, ff, nd, drop, ml, cfg['pad_id'])


class _CustomModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _CustomModelWrapper(nn.Module):
    """Wraps BabyLMModel to expose a HF-like causal-LM interface (.logits)."""
    def __init__(self, model, pad_id=0):
        super().__init__()
        self.m = model
        self.pad_id = pad_id

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        src_key_padding_mask = ~attention_mask.bool()
        enc_out = self.m.encoder(input_ids, src_key_padding_mask=src_key_padding_mask)
        memory = self.m.prob_encoder(enc_out)
        seq_len = input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        logits = self.m.decoder(
            input_ids, memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return _CustomModelOutput(logits)

    def resize_token_embeddings(self, new_size):
        pass  # no-op for custom model


def _is_custom_checkpoint(path):
    return bool(path) and os.path.isfile(os.path.join(path, 'best_model.pt'))


def _load_custom_model(path):
    ckpt = torch.load(os.path.join(path, 'best_model.pt'), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    base = _BabyLMModel(cfg)
    base.load_state_dict(ckpt['model_state_dict'])
    base.eval()
    return _CustomModelWrapper(base, pad_id=cfg['pad_id']), cfg


# ─────────────────────────────────────────────────────────────────────────────

def tokenize_encoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i], add_special_tokens=False)
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i], add_special_tokens=False)
        batch["good_inputs"].append(good_tokens)
        batch["bad_inputs"].append(bad_tokens)
        batch["good_labels"].append(good_tokens)
        batch["bad_labels"].append(bad_tokens)

    return batch

def tokenize_decoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i])
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i])
        batch["good_inputs"].append(good_tokens[:-1])
        batch["bad_inputs"].append(bad_tokens[:-1])
        batch["good_labels"].append(good_tokens[1:])
        batch["bad_labels"].append(bad_tokens[1:])

    return batch


def padding_collate_fn(batch, max_len=1024, left_padding=False, pad_token_id=0):
    """ 
        Pads each list with pad_token_id and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        if "labels" in key:
            padded_batch[key] = torch.full((len(batch), largest), -100, dtype=torch.long)
        else:
            padded_batch[key] = torch.full((len(batch), largest), pad_token_id, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def evaluate_decoder(model, dataloader, tokenizer, device):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            good_inputs = batch['good_inputs'].to(device=device)
            bad_inputs = batch['bad_inputs'].to(device=device)
            good_labels = batch['good_labels'].to(device=device)
            bad_labels = batch['bad_labels'].to(device=device)

            good_outputs = model(good_inputs, attention_mask=good_inputs != pad_token_id).logits
            bad_outputs = model(bad_inputs, attention_mask=bad_inputs != pad_token_id).logits
            good_loss = loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))
            bad_loss = loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_loss = good_loss.view(good_inputs.shape[0], -1).mean(dim=1)
            bad_loss = bad_loss.view(bad_inputs.shape[0], -1).mean(dim=1)
            for b in range(len(good_loss)):
                if good_loss[b] < bad_loss[b]:
                    correct += 1
                total += 1
    return correct / total


def evaluate_encoder(model, dataloader, tokenizer, device):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            good_loss = 0.0
            bad_loss = 0.0

            good_inputs = batch['good_inputs'].to(device=device)
            bad_inputs = batch['bad_inputs'].to(device=device)
            good_labels = batch['good_labels'].to(device=device)
            bad_labels = batch['bad_labels'].to(device=device)

            max_len = batch['good_inputs'].shape[1]
            for i in range(max_len):
                masked_good_inputs = good_inputs.clone()
                masked_good_inputs[:, i] = tokenizer.mask_token_id
                good_outputs = model(masked_good_inputs, attention_mask=good_inputs != pad_token_id).logits
                good_loss += loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))

            max_len = batch['bad_inputs'].shape[1]
            for i in range(max_len):
                masked_bad_inputs = bad_inputs.clone()
                masked_bad_inputs[:, i] = tokenizer.mask_token_id
                bad_outputs = model(masked_bad_inputs, attention_mask=bad_inputs != pad_token_id).logits
                bad_loss += loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_loss = good_loss.view(good_inputs.shape[0], -1).mean(dim=1)
            bad_loss = bad_loss.view(bad_inputs.shape[0], -1).mean(dim=1)

            for b in range(len(good_loss)):
                if good_loss[b] < bad_loss[b]:
                    correct += 1
                total += 1
    return correct / total

def discover_subsets(data_path):
    """Discover all .jsonl files in data_path and return their basenames."""
    jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
    subsets = [os.path.splitext(os.path.basename(f))[0] for f in jsonl_files]
    return sorted(subsets)


def load_tokenizer(args):
    """Load tokenizer with SPM support if needed."""
    # Determine SPM path
    spm_path = args.spm_path
    if not spm_path and args.model_path:
        for _fname in ('spm.model', 'spm_model.model'):
            _candidate = os.path.join(args.model_path, _fname)
            if os.path.isfile(_candidate):
                spm_path = _candidate
                break

    # For custom .pt checkpoints, use the lightweight SPM wrapper directly
    if spm_path and _is_custom_checkpoint(args.model_path):
        ckpt = torch.load(
            os.path.join(args.model_path, 'best_model.pt'),
            map_location='cpu', weights_only=False,
        )
        cfg = ckpt.get('config', {})
        print(f"Loading SPMTokenizerWrapper from {spm_path}")
        return _SPMTokenizerWrapper(
            spm_path,
            bos_id=cfg.get('bos_id', 2),
            eos_id=cfg.get('eos_id', 3),
            pad_id=cfg.get('pad_id', 0),
        )

    # Load tokenizer
    if spm_path:
        print(f"Loading DebertaV2Tokenizer from {spm_path}")
        tokenizer = DebertaV2Tokenizer(vocab_file=spm_path)
    else:
        print(f"Loading AutoTokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Ensure special tokens exist
    if tokenizer.pad_token is None:
        print("Adding [PAD] token")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.bos_token is None:
        print("Adding <s> token")
        tokenizer.add_special_tokens({'bos_token': '<s>'})
    if tokenizer.eos_token is None:
        print("Adding </s> token")
        tokenizer.add_special_tokens({'eos_token': '</s>'})

    return tokenizer


def main():
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set default model if not specified
    model_name = "prajjwal1/bert-tiny" if args.model_type == "encoder" else "sshleifer/tiny-gpt2"
    if not args.model_path:
        args.model_path = model_name
    
    # Discover subsets
    if args.subsets:
        subsets = [s.strip() for s in args.subsets.split(',')]
        print(f"Evaluating {len(subsets)} specified subsets: {subsets[:5]}{'...' if len(subsets) > 5 else ''}")
    else:
        subsets = discover_subsets(args.data_path)
        print(f"Auto-discovered {len(subsets)} subsets: {subsets[:5]}{'...' if len(subsets) > 5 else ''}")
    
    if not subsets:
        print("No subsets found. Exiting.")
        return
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    tokenize_fn = partial(tokenize_encoder if args.model_type == "encoder" else tokenize_decoder, tokenizer=tokenizer)
    
    # Load model
    if _is_custom_checkpoint(args.model_path):
        model, _ = _load_custom_model(args.model_path)
    elif args.model_type == "encoder":
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Prepare evaluation function and collate function
    evaluate_fn = evaluate_encoder if args.model_type == "encoder" else evaluate_decoder
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = partial(padding_collate_fn, pad_token_id=pad_token_id)
    
    results = {}
    print("\nEvaluating subsets...")
    for subset in tqdm(subsets):
        subset_path = os.path.join(args.data_path, f'{subset}.jsonl')
        
        # Check if file exists
        if not os.path.isfile(subset_path):
            print(f"Warning: {subset_path} not found. Skipping.")
            continue
        
        # Load dataset
        try:
            dataset = datasets.load_dataset('json', data_files={'train': subset_path}, split='train')
        except Exception as e:
            print(f"Warning: Failed to load {subset_path}: {e}. Skipping.")
            continue
        
        # Validate columns
        if 'sentence_good' not in dataset.column_names or 'sentence_bad' not in dataset.column_names:
            print(f"Warning: {subset} missing required columns. Has: {dataset.column_names}. Skipping.")
            continue
        
        # Tokenize and evaluate
        try:
            dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset.column_names)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            accuracy = evaluate_fn(model, dataloader, tokenizer, device)
            results[subset] = accuracy
            print(f"  {subset}: {len(dataset)} examples, accuracy = {accuracy:.4f}")
        except Exception as e:
            print(f"Warning: Error evaluating {subset}: {e}. Skipping.")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    for subset, accuracy in results.items():
        print(f"  {subset}: {accuracy:.4f}")
    
    if results:
        macro_avg = sum(results.values()) / len(results)
        print("="*60)
        print(f"Macro Average: {macro_avg:.4f}")
        print("="*60)
    else:
        print("No subsets were successfully evaluated.")


if __name__ == '__main__':
    main()