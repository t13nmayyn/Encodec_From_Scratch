import torch
from torch import nn
from torchaudio import transforms
from torch.nn.utils.parametrizations import weight_norm


class NConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=(1, 1), dilation=(1, 1), padding=None):
        super().__init__()
        if padding is None:
            padding = (dilation[0] * (kernel[0] - 1) // 2,
                       dilation[1] * (kernel[1] - 1) // 2)
        self.conv = weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation))

    def forward(self, x):
        return self.conv(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length=None, channels=32, kernel=(3, 9), stride=(1, 2)):
        super().__init__()
        hop_length = hop_length or n_fft // 4
        self.spec = transforms.Spectrogram(
            n_fft=n_fft, win_length=n_fft, hop_length=hop_length,
            power=None, normalized=True, center=False)
        self.layers = nn.ModuleList([
            NConv2d(2, channels, kernel),
            NConv2d(channels, channels, kernel, stride=stride, dilation=(1, 1)),
            NConv2d(channels, channels, kernel, stride=stride, dilation=(2, 1)),
            NConv2d(channels, channels, kernel, stride=stride, dilation=(4, 1)),
            NConv2d(channels, channels, (3, 3)),
        ])
        self.act = nn.LeakyReLU(0.2)
        self.conv_final = NConv2d(channels, 1, kernel=(3, 3))

    def forward(self, x):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)                       # (B, samples)
        z = self.spec(x)                           # (B, F, T) complex
        z = torch.stack([z.real, z.imag], dim=1)   # (B, 2, F, T)
        z = z.transpose(-1, -2)                    # (B, 2, T, F)  keep this transpose
        feats = []
        for layer in self.layers:
            z = self.act(layer(z))
            feats.append(z)
        return self.conv_final(z), feats


class MultiSTFTDiscriminator(nn.Module):
    def __init__(self, n_ffts=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.discs = nn.ModuleList([STFTDiscriminator(n) for n in n_ffts])

    def forward(self, x):
        return [d(x) for d in self.discs]


if __name__ == "__main__":
    d = STFTDiscriminator(512)
    logits, feats = d(torch.randn(2, 24000))
    print(logits.shape, [f.shape for f in feats])   # time axis should be equal everywhere
