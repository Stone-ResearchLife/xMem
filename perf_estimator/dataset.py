import torchvision
import torchaudio
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
from pathlib import Path
from torchaudio.transforms import Resample, MelSpectrogram


def image_dataset(batch: int = 200, image_size: Tuple[int, int] = (86, 86), float16: bool = False) -> torch.utils.data.DataLoader:

    transform_option = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ]
    if float16:
        transform_option.append(torchvision.transforms.Lambda(lambda x: x.half()))
    transform = torchvision.transforms.Compose(transform_option)
    # temp_dataset_dir = temp_dir_with_specific_path("pytorch", "datasets")
    temp_dataset_dir = os.path.join(Path().home(), 'pytorch_datasets')
    train_data = torchvision.datasets.CIFAR10(root=temp_dataset_dir, train=True, download=True, transform=transform)
    # train_data = torchvision.datasets.Food101(root=temp_dataset_dir, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)


def audio_dataset(batch_size: int = 32, sample_rate: int = 16000, n_mels: int = 80, float16: bool = False):
    temp_dataset_dir = os.path.join(Path().home(), 'pytorch_datasets')

    train_data = torchaudio.datasets.LIBRISPEECH(root=temp_dataset_dir, download=True, url="train-clean-100")

    resampler = Resample(orig_freq=16000, new_freq=sample_rate)
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    def build_vocab(dataset):
        all_text = ''
        for i in range(len(dataset)):
            _, _, transcript, *rest = dataset[i]
            all_text += transcript.lower()
        vocab = sorted(set(all_text))
        char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # 从1开始编码，0留给blank
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        return char_to_idx, idx_to_char

    char_to_idx, idx_to_char = build_vocab(train_data)
    vocab_size = len(char_to_idx) + 1

    class TransformLibriSpeech(torch.utils.data.Dataset):
        def __init__(self, dataset, resampler, mel_transform, float16, char_to_idx):
            self.dataset = dataset
            self.resampler = resampler
            self.mel_transform = mel_transform
            self.float16 = float16
            self.char_to_idx = char_to_idx

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            waveform, sample_rate, transcript, *rest = self.dataset[idx]
            waveform = self.resampler(waveform)
            mel_spec = self.mel_transform(waveform)
            if self.float16:
                mel_spec = mel_spec.half()
            seq_length = mel_spec.size(-1)  # 时间维度

            transcript = transcript.lower()
            target = [self.char_to_idx.get(c, 0) for c in transcript]
            target_length = len(target)

            return mel_spec.squeeze(0).transpose(1, 0), seq_length, torch.tensor(target, dtype=torch.long), target_length

    def collate_fn(batch):
        inputs = [item[0] for item in batch]  # (时间, 特征)
        input_lengths = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        targets = [item[2] for item in batch]
        target_lengths = torch.tensor([item[3] for item in batch], dtype=torch.int64)

        inputs_padded = pad_sequence(inputs, batch_first=True)
        targets_padded = torch.cat(targets)

        return inputs_padded, input_lengths, targets_padded, target_lengths

    transformed_data = TransformLibriSpeech(train_data, resampler, mel_transform, float16, char_to_idx)
    return torch.utils.data.DataLoader(transformed_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), vocab_size




