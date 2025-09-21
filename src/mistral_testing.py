import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import json
import numpy as np


def calculate_token_confidence(logits, token_ids, tokenizer):
    """
    Calculate confidence score based on token probabilities.
    
    Args:
        logits: Model logits for generated tokens [seq_len, batch_size, vocab_size]
        token_ids: Generated token IDs [seq_len]
        tokenizer: Tokenizer for decoding tokens
    
    Returns:
        dict: Confidence metrics including per-token details
    """
    # Handle different logits dimensions
    if logits.dim() == 3:
        # Shape: [seq_len, batch_size, vocab_size] -> squeeze batch dimension
        logits = logits.squeeze(1)  # [seq_len, vocab_size]
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # [seq_len, vocab_size]
    
    # Ensure we don't exceed the available sequence length
    seq_len = min(len(probs), len(token_ids))
    
    # Get probabilities and decode tokens for analysis
    token_details = []
    token_probs = []
    
    for i in range(seq_len):
        token_id = token_ids[i].item() if hasattr(token_ids[i], 'item') else token_ids[i]
        
        # Make sure token_id is within vocabulary bounds
        if 0 <= token_id < probs.shape[-1]:
            token_prob = probs[i][token_id].item()
            token_text = tokenizer.decode([token_id])
            
            # Calculate entropy for this position
            prob_dist = torch.clamp(probs[i], min=1e-10)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist)).item()
            
            token_details.append({
                'token_id': token_id,
                'token_text': token_text,
                'confidence': token_prob,
                'entropy': entropy,
                'position': i
            })
            token_probs.append(token_prob)
    
    if not token_probs:
        return {
            'mean_confidence': 0.0,
            'min_confidence': 0.0,
            'confidence_std': 0.0,
            'entropy': 0.0,
            'token_details': [],
            'num_tokens': 0
        }
    
    # Calculate overall metrics
    mean_confidence = np.mean(token_probs)
    min_confidence = np.min(token_probs)
    confidence_std = np.std(token_probs)
    mean_entropy = np.mean([t['entropy'] for t in token_details])
    
    return {
        'mean_confidence': float(mean_confidence),
        'min_confidence': float(min_confidence),
        'confidence_std': float(confidence_std),
        'entropy': float(mean_entropy),
        'token_details': token_details,
        'num_tokens': len(token_probs)
    }


def extract_domain_confidence(token_details, generated_text):
    """
    Extract confidence scores for individual domain names from token details.
    
    Args:
        token_details: List of token details from calculate_token_confidence
        generated_text: Full generated text
    
    Returns:
        list: Confidence scores for each domain found
    """
    domain_confidences = []
    
    # Find domain patterns in the generated text
    import re
    domain_pattern = r'"([a-zA-Z0-9\-]+\.com)"'
    domains = re.findall(domain_pattern, generated_text)
    
    if not domains:
        return []
    
    # Reconstruct full text with token positions
    full_text = ""
    token_positions = {}
    
    for i, token_detail in enumerate(token_details):
        token_start = len(full_text)
        token_text = token_detail['token_text']
        full_text += token_text
        token_end = len(full_text)
        
        # Map character positions to token details
        for char_pos in range(token_start, token_end):
            token_positions[char_pos] = token_detail
    
    # Find each domain's confidence
    for domain in domains:
        domain_start = generated_text.find(f'"{domain}"')
        if domain_start != -1:
            # Find tokens that contribute to this domain name (excluding quotes)
            domain_name = domain  # just the domain without quotes
            domain_name_start = domain_start + 1  # skip opening quote
            domain_name_end = domain_name_start + len(domain_name)
            
            # Get confidence scores for tokens in this domain
            domain_token_confidences = []
            domain_token_entropies = []
            
            for char_pos in range(domain_name_start, domain_name_end):
                if char_pos in token_positions:
                    token_detail = token_positions[char_pos]
                    if token_detail['confidence'] not in [t['confidence'] for t in domain_token_confidences]:
                        domain_token_confidences.append(token_detail)
                        domain_token_entropies.append(token_detail['entropy'])
            
            if domain_token_confidences:
                # Calculate domain-specific metrics
                confidences = [t['confidence'] for t in domain_token_confidences]
                entropies = [t['entropy'] for t in domain_token_confidences]
                
                domain_confidence = {
                    'domain': domain,
                    'mean_confidence': float(np.mean(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences)),
                    'confidence_std': float(np.std(confidences)),
                    'mean_entropy': float(np.mean(entropies)),
                    'num_tokens': len(confidences),
                    'overall_score': float(np.mean(confidences) * (1 - np.mean(entropies) / 10))
                }
                domain_confidences.append(domain_confidence)
    
    return domain_confidences


def generate_domains_prompt(description: str) -> str:    
    return f"""You are a creative domain name generator. Generate domain names for this business.
    BUSINESS DESCRIPTION:
    {description}
    DOMAIN REQUIREMENTS:
    - Generate exactly 3 domain suggestions
    - Use .com extension only
    - 6-15 characters (excluding .com)
    - Memorable, brandable, easy to spell
    - Relevant to business but not too literal
    - Avoid hyphens, numbers, or complex words
    FORMAT YOUR RESPONSE EXACTLY IN JSON, LIKE THIS:
    {{
    "suggestions": [
        {{"domain": "firstdomain.com"}},
        {{"domain": "seconddomain.com"}},
        {{"domain": "thirddomain.com"}}
    ]
    }}
    OUTPUT:
    """

def generate_domains_with_confidence_scores(model, tokenizer, description: str, max_new_tokens: int = 80, temperature : int = 0.2) -> dict:
    """
    Generate domain name suggestions given a business description.
    Returns the response with confidence scores.
    """
    prompt = generate_domains_prompt(description)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        # Generate with return_dict_in_generate=True to get logits
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated tokens (excluding input prompt)
    generated_tokens = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Calculate confidence scores
    try:
        if hasattr(outputs, 'scores') and outputs.scores:
            # Stack all logits from generation steps
            all_logits = torch.stack(outputs.scores, dim=0)  # [seq_len, vocab_size]
            
            # Debug information
            print(f"Debug - Logits shape: {all_logits.shape}")
            print(f"Debug - Generated tokens shape: {generated_tokens.shape}")
            print(f"Debug - Vocab size: {all_logits.shape[-1]}")
            print(f"Debug - Max token ID: {generated_tokens.max().item()}")
            print(f"Debug - Min token ID: {generated_tokens.min().item()}")
            
            confidence_metrics = calculate_token_confidence(all_logits, generated_tokens, tokenizer)
            
            # Extract per-domain confidence scores
            domain_confidences = extract_domain_confidence(confidence_metrics['token_details'], generated_text)
            
        else:
            print("Warning: No scores available from model generation")
            confidence_metrics = {
                'mean_confidence': 0.0,
                'min_confidence': 0.0,
                'confidence_std': 0.0,
                'entropy': 0.0,
                'token_details': [],
                'num_tokens': 0
            }
            domain_confidences = []
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        confidence_metrics = {
            'mean_confidence': 0.0,
            'min_confidence': 0.0,
            'confidence_std': 0.0,
            'entropy': 0.0,
            'token_details': [],
            'num_tokens': 0,
            'error': str(e)
        }
        domain_confidences = []
    
    # Try to parse the generated JSON
    try:
        # Extract JSON part from the generated text
        start_idx = generated_text.find('{')
        end_idx = generated_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_text = generated_text[start_idx:end_idx]
            parsed_domains = json.loads(json_text)
        else:
            parsed_domains = {"error": "No valid JSON found in response"}
    except json.JSONDecodeError:
        parsed_domains = {"error": "Failed to parse JSON response"}
    
    return {
        'domains': parsed_domains,
        'raw_response': generated_text,
        'confidence': confidence_metrics,
        'domain_confidences': domain_confidences,  # NEW: Per-domain confidence scores
        'overall_confidence_score': confidence_metrics['mean_confidence'] * (1 - confidence_metrics['entropy'] / 10)  # Combined score
    }

def generate_domains(model, tokenizer, description: str, max_new_tokens: int = 80) -> str:
    """
    Generate domain name suggestions given a business description.
    Returns the raw JSON string from the model.
    """
    prompt = f"""You are a creative domain name generator. Generate domain names for this business.
BUSINESS DESCRIPTION:
{description}
DOMAIN REQUIREMENTS:
- Generate exactly 3 domain suggestions
- Use .com extension only
- 6-15 characters (excluding .com)
- Memorable, brandable, easy to spell
- Relevant to business but not too literal
- Avoid hyphens, numbers, or complex words
FORMAT YOUR RESPONSE EXACTLY IN JSON, LIKE THIS:
{{
  "suggestions": [
    {{"domain": "firstdomain.com"}},
    {{"domain": "seconddomain.com"}},
    {{"domain": "thirddomain.com"}}
  ]
}}
OUTPUT:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.01
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)