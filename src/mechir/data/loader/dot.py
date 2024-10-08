from . import pad
import torch

class DotDataCollator:
    def __init__(self, 
                 tokenizer, 
                 transformation_func : callable,
                 special_mask=False,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.transformation_func = transformation_func
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_mask = special_mask
        self.special_token = "[X]"

        # add special token (pad w/attn) to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[X]"]})

    # Helper function to align and insert special tokens to the original document
    # NOTE: for now, this assumes tokens are only being added to the end
    def align_and_insert_special_tokens(self, original_ids, perturbed_ids):    
        # Get the SEP token ID
        sep_token_id = self.tokenizer.sep_token_id

        # Initialize a list to hold the aligned sequences
        aligned_ids = []

        # Iterate through each original and perturbed ID sequence
        for orig_seq, pert_seq in zip(original_ids, perturbed_ids):
            # Calculate lengths
            orig_length = orig_seq.size(0)
            pert_length = pert_seq.size(0)

            # Calculate how many special tokens to append
            length_diff = pert_length - orig_length

            # Clone the original sequence to avoid modifying it
            aligned_seq = orig_seq.clone()

            # Insert special tokens before the SEP token if needed
            if length_diff > 0:
                special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)
                special_tokens_tensor = torch.tensor([special_token_id] * length_diff, dtype=torch.long, device=orig_seq.device)

                # Find the index of the SEP token (which is the last token)
                sep_index = orig_length - 1  # Last token is the SEP token

                # Prepare the new aligned tensor
                aligned_seq = torch.cat((aligned_seq[:sep_index], special_tokens_tensor, aligned_seq[sep_index:]), dim=0)

            # Append the aligned sequence to the list
            aligned_ids.append(aligned_seq)

        return aligned_ids
    
    
    def pad_documents(self, documents, max_length):
        padded_docs, attention_masks = [], []
        
        for doc in documents:
            # Calculate how much padding is needed
            padding_length = max_length - doc.size(0) 

            # Calculate padding for original document
            attention_mask = [1] *  doc.size(0)
            
            if padding_length > 0:
                padding_tensor = torch.full((padding_length,), self.tokenizer.pad_token_id)  # Shape: (1, padding_length)
                padded_doc = torch.cat((doc, padding_tensor), dim=0)
                attention_mask += [0] * padding_length
            else:
                padded_doc = doc # already a tensor
                
            padded_docs.append(padded_doc)
            attention_masks.append(torch.tensor(attention_mask))

        return torch.stack(padded_docs), torch.stack(attention_masks)

    def __call__(self, batch) -> dict:
        tokenized_queries = self.tokenizer(
            [query for query, _ in batch],
            padding=True,
            truncation=False,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )

        # (1) Create perturbed pairs
        batch_docs = [doc for _, doc in batch]
        batch_perturbed_docs = [self.transformation_func(doc, query=query) for query, doc in batch]
        
        # (2) Tokenize original and perturbed documents without padding
        tokenized_docs_input_ids = [self.tokenizer(doc, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0) for doc in batch_docs]
        tokenized_perturbed_docs_input_ids = [self.tokenizer(doc, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0) for doc in batch_perturbed_docs]

        # (3) Identify where the perturbation occurred and align document pairs (options: injection, deletion, replacement)
        # NOTE: only injection at end is coded right now
        aligned_baseline_docs = self.align_and_insert_special_tokens(
            tokenized_docs_input_ids,
            tokenized_perturbed_docs_input_ids
        )
        
        # (4) Find the maximum length across both original and perturbed documents
        max_length = max(max(ids.size(0) for ids in aligned_baseline_docs), max(ids.size(0) for ids in tokenized_perturbed_docs_input_ids))

        # (5) Pad all document in batch with regular [PAD] token
        padded_baseline_docs, padded_baseline_attn_masks = self.pad_documents(aligned_baseline_docs, max_length)
        padded_perturbed_docs, padded_perturbed_attn_masks = self.pad_documents(tokenized_perturbed_docs_input_ids, max_length)

        # (6) Format input_ids and attention_masks
        final_tokenized_docs = {"input_ids": padded_baseline_docs, "attention_mask": padded_baseline_attn_masks} 
        final_tokenized_perturbed_docs = {"input_ids": padded_perturbed_docs, "attention_mask": padded_perturbed_attn_masks} 

        return {
            "queries": dict(tokenized_queries),
            "documents": dict(final_tokenized_docs),
            "perturbed_documents": dict(final_tokenized_perturbed_docs),
        }

__all__ = ["DotDataCollator"]