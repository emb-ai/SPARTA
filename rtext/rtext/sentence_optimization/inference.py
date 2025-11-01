import torch
from lisa.utils.utils import dict_to_cuda


def inference(
    sample: dict, 
    model: torch.nn.Module, 
    original_context: dict = None, 
    adv_embeds: torch.Tensor = None, 
    adv_ids: torch.Tensor = None,
    device='cuda'
    ) -> dict:
    """
    Perform inference with adversarial embeddings or adversarial IDs.
    """
    sample.pop('inputs_embeds', None)
    
    if adv_ids is not None:
        sample = insert_adv_ids(
            sample, 
            original_context, 
            adv_ids, 
            device=device)

    sample = dict_to_cuda(sample, device=device)
    sample['images'] = sample['images'].half()
    sample['images_clip'] = sample['images_clip'].half()

    if adv_embeds is not None:
        sample['inputs_embeds'] = model.prepare_inputs_labels_for_multimodal(
            sample['input_ids'],
            sample['attention_masks'],
            None,
            sample['labels'],
            sample['images_clip'])[3].detach()

        sample['inputs_embeds'] = insert_adv_embeds(
            adv_embeds,
            original_context, 
            device=device)

        assert sample['input_ids'].shape[1] + model.insert_tokens == sample['inputs_embeds'].shape[1], \
            'input_ids and inputs_embeds should have the same length'
            
    with torch.amp.autocast('cuda'):
        output_dict = model(**sample)
    
    return output_dict
    

def insert_adv_embeds(
    adv_embeds: torch.Tensor,
    original_context: dict, 
    device: str = 'cuda'
    ) -> torch.Tensor:
    """
    Insert adversarial embeddings into the input embeddings.
    Optimize only those adversarial embeddings.
    """
    inputs_embeds = torch.cat((
        original_context['prefix_embeds'].detach(),
        adv_embeds.to(device),
        original_context['suffix_embeds'].detach())
    ).unsqueeze(0).half()

    return inputs_embeds


def insert_adv_ids(
    input_dict: dict, 
    original_context: dict, 
    adv_ids: torch.Tensor,
    device: str = 'cuda'
    ) -> dict:
    """
    Insert adversarial IDs into the input dictionary. 
    Pad attention mask and labels to match the size of input_ids
    """
    input_dict['input_ids'] = torch.cat([
        original_context['prefix_ids'],
        adv_ids.to(device),
        original_context['suffix_ids']
    ]).unsqueeze(0).int()
    
    input_dict['attention_masks'] = torch.cat([
        original_context['prefix_attn'],
        torch.ones_like(adv_ids, dtype=torch.bool).to(device),
        original_context['suffix_attn']
    ]).unsqueeze(0)
    
    input_dict['labels'] = adjust_tensor_size(
        input_dict['labels'][0], 
        input_dict['input_ids'].size()[1]
    ).unsqueeze(0)

    return input_dict


def adjust_tensor_size(
    tensor: torch.Tensor, 
    target_size: int, 
    pad_value: int = -100
    ) -> torch.Tensor:
    """
    Adjust the size of a tensor to match the target size, padding or truncating as necessary.
    """
    current_size = tensor.size(0)
    
    if current_size > target_size:
        truncated_tensor = tensor[-target_size:]
    else:
        padding_size = target_size - current_size
        padding = torch.full((padding_size,), pad_value)
        truncated_tensor = torch.cat((padding, tensor.cpu()), dim=0)
    return truncated_tensor