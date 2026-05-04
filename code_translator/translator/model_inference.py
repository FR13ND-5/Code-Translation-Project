import os
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from .model import Seq2Seq

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BASE DIR (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Correct path to your model
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "checkpoint-best-bleu", "pytorch_model.bin")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load model config
config = RobertaConfig.from_pretrained("microsoft/codebert-base")
encoder = RobertaModel.from_pretrained("microsoft/codebert-base", config=config)

import torch.nn as nn
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    config=config,
    beam_size=10,
    max_length=256,
    sos_id=tokenizer.cls_token_id,
    eos_id=tokenizer.sep_token_id
)

# LOAD YOUR BEST BLEU MODEL
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def translate_code(code):
    tokens = tokenizer.tokenize(code)[:254]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids += [tokenizer.pad_token_id] * (256 - len(ids))

    source_ids = torch.tensor([ids]).to(device)
    source_mask = source_ids.ne(tokenizer.pad_token_id)

    with torch.no_grad():
        preds = model(source_ids=source_ids, source_mask=source_mask)

    pred = preds[0][0].cpu().numpy().tolist()

    if 0 in pred:
        pred = pred[:pred.index(0)]

    return tokenizer.decode(pred, clean_up_tokenization_spaces=False)