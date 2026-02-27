"""
Pretrained audio encoder wrappers.

Six encoders spanning semantic, signal-level, and self-supervised paradigms.
Each encoder implements .encode(wav, sr) -> np.ndarray.
"""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


class BaseEncoder:
    """Common interface for all audio encoders."""
    def __init__(self, device="cuda"):
        self.device = device
        self.target_sr: int = 16000
        self.name: str = "base"

    def encode(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        """Return (T', D) frame-level or (D,) global-level embedding."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# 1. Whisper large-v3  (ASR, 16 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class WhisperEncoder(BaseEncoder):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 16000
        self.name = "whisper"
        from transformers import WhisperModel, WhisperFeatureExtractor
        model_id = "openai/whisper-large-v3"
        logger.info(f"Loading Whisper: {model_id}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).encoder
        self.model = self.model.to(device).eval().half()

    @torch.no_grad()
    def encode(self, wav, sr=16000):
        audio_np = wav.squeeze().numpy()
        inputs = self.feature_extractor(
            audio_np, sampling_rate=sr, return_tensors="pt"
        )
        feats = inputs.input_features.to(self.device).half()
        outputs = self.model(feats)
        return outputs.last_hidden_state.squeeze(0).float().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLAP  (Contrastive, 48 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class CLAPEncoder(BaseEncoder):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 48000
        self.name = "clap"
        import laion_clap
        logger.info("Loading CLAP (LAION-630k)")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()

    @torch.no_grad()
    def encode(self, wav, sr=48000):
        audio_np = wav.squeeze().numpy().astype(np.float32)
        embed = self.model.get_audio_embedding_from_data(
            x=[audio_np], use_tensor=False
        )
        return embed.squeeze()  # (512,)


# ═══════════════════════════════════════════════════════════════════════════
# 3. VGGish  (Classification, 16 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class VGGishEncoder(BaseEncoder):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 16000
        self.name = "vggish"
        logger.info("Loading VGGish (AudioSet)")
        self.model = torch.hub.load(
            "harritaylor/torchvggish", "vggish", trust_repo=True
        )
        self.model.postprocess = False
        self.model.eval()
        self.model.to(device)
        from torchvggish import vggish_input
        self._preprocess = vggish_input.waveform_to_examples

    @torch.no_grad()
    def encode(self, wav, sr=16000):
        audio_np = wav.squeeze().numpy()
        examples = self._preprocess(audio_np, sr, return_tensor=True)
        examples = examples.to(self.device)
        embeddings = self.model.forward(examples)
        return embeddings.cpu().numpy()  # (N, 128)


# ═══════════════════════════════════════════════════════════════════════════
# 4. EnCodec 24 kHz  (Neural codec, 24 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class EnCodecEncoder(BaseEncoder):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 24000
        self.name = "encodec"
        from encodec import EncodecModel
        logger.info("Loading EnCodec 24kHz")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode(self, wav, sr=24000):
        x = wav.unsqueeze(0).to(self.device)    # (1, 1, T)
        emb = self.model.encoder(x)             # (1, 128, T')
        return emb.squeeze(0).permute(1, 0).cpu().numpy()  # (T', 128)


# ═══════════════════════════════════════════════════════════════════════════
# 5. AudioMAE  (Masked reconstruction, 16 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class _Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=(1024, 128), patch_size=(16, 16),
                 in_chans=1, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _AudioMAEViT(nn.Module):
    """Minimal ViT-Base encoder matching AudioMAE checkpoint."""
    def __init__(self, img_size=(1024, 128), patch_size=(16, 16),
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, 1, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]  # patch embeddings only


class AudioMAEEncoder(BaseEncoder):
    CKPT_ID = "1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwQ"
    MEAN, STD = -4.2677393, 4.5689974

    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 16000
        self.name = "audiomae"
        logger.info("Loading AudioMAE (ViT-Base, AudioSet)")
        self.model = _AudioMAEViT().to(device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self):
        from config import CACHE_DIR
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CACHE_DIR / "audiomae_pretrained.pth"
        if not ckpt_path.exists():
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={self.CKPT_ID}"
                logger.info("Downloading AudioMAE checkpoint ...")
                gdown.download(url, str(ckpt_path), quiet=False)
            except Exception as e:
                logger.warning(f"Cannot download AudioMAE ckpt: {e}. "
                               "Using random init.")
                return
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        enc_keys = {k: v for k, v in model_state.items()
                    if not k.startswith(("decoder", "mask_token", "decoder_"))}
        missing, unexpected = self.model.load_state_dict(enc_keys, strict=False)
        logger.info(f"AudioMAE loaded — missing {len(missing)}, "
                    f"unexpected {len(unexpected)} keys")

    def _mel(self, wav):
        wav_1d = wav.squeeze(0)
        if wav_1d.ndim == 0 or wav_1d.shape[0] < 400:
            wav_1d = F.pad(wav_1d, (0, 400 - wav_1d.shape[0]))
        fbank = torchaudio.compliance.kaldi.fbank(
            wav_1d.unsqueeze(0), htk_compat=True,
            sample_frequency=self.target_sr, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.0,
            frame_shift=10,
        )
        fbank = (fbank - self.MEAN) / (self.STD * 2)
        target = 1024
        if fbank.shape[0] < target:
            fbank = F.pad(fbank, (0, 0, 0, target - fbank.shape[0]))
        else:
            fbank = fbank[:target]
        return fbank.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def encode(self, wav, sr=16000):
        mel = self._mel(wav).to(self.device)
        out = self.model(mel)
        return out.squeeze(0).cpu().numpy()  # (512, 768)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Wav2Vec 2.0 Base  (Self-supervised, 16 kHz)
# ═══════════════════════════════════════════════════════════════════════════

class Wav2VecEncoder(BaseEncoder):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.target_sr = 16000
        self.name = "wav2vec"
        from transformers import Wav2Vec2Model
        model_id = "facebook/wav2vec2-base"
        logger.info(f"Loading Wav2Vec2: {model_id}")
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.model = self.model.to(device).eval().half()

    @torch.no_grad()
    def encode(self, wav, sr=16000):
        x = wav.squeeze().to(self.device).half()
        outputs = self.model(x.unsqueeze(0))
        return outputs.last_hidden_state.squeeze(0).float().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

ENCODER_REGISTRY = {
    "whisper":  WhisperEncoder,
    "clap":     CLAPEncoder,
    "vggish":   VGGishEncoder,
    "encodec":  EnCodecEncoder,
    "audiomae": AudioMAEEncoder,
    "wav2vec":  Wav2VecEncoder,
}
