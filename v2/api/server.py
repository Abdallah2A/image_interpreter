from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
from transformers import AutoTokenizer
from steps.model_definition import VisionEncoderDecoder

app = FastAPI()

image_size = 128
hidden_size = 192
num_layers = (6, 6)
num_heads = 8
patch_size = 8
channels_in = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
num_emb = tokenizer.vocab_size

model = VisionEncoderDecoder(image_size=image_size, channels_in=channels_in,
                             num_emb=num_emb, patch_size=patch_size,
                             num_layers=num_layers, hidden_size=hidden_size,
                             num_heads=num_heads).to(device)

model.load_state_dict(torch.load("model/final_model_v2.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def generate_caption(model, image, tokenizer, max_len=50):
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        encoded_seq = model.encoder(image)

        seq = torch.tensor([[101]], device=device)  # [CLS]
        generated_tokens = []

        for _ in range(max_len):
            padded_seq = F.pad(seq, (0, max_len - seq.shape[1]), value=0)
            padding_mask = torch.ones(1, seq.shape[1], device=device).long()
            padding_mask = F.pad(padding_mask, (0, max_len - seq.shape[1]), value=0)
            bool_padding_mask = (padding_mask == 0)

            decoded_seq = model.decoder(padded_seq, encoded_seq,
                                        input_padding_mask=bool_padding_mask)
            next_token_logits = decoded_seq[:, seq.shape[1]-1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token_id)
            if next_token_id == 102:  # [SEP]
                break
            seq = torch.cat([seq, torch.tensor([[next_token_id]], device=device)], dim=1)

        caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return caption


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    caption = generate_caption(model, image, tokenizer)
    return JSONResponse(content={"caption": caption})
