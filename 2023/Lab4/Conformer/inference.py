import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from src.encoder import ConformerBlock

class Classifier(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_spks: int = 600,
        nhead: int = 16,
        dropout: float = 0.1,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.dropout = nn.Dropout(dropout)

        # Replace TransformerEncoderLayer with ConformerBlock
        self.encoder_layer = ConformerBlock(
            encoder_dim=d_model,
            num_attention_heads=nhead,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )

        # Project the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels: Tensor) -> Tensor:
        """
        args:
            mels: (batch_size, length, 40)
        return:
            out: (batch_size, n_spks)
        """
        # out: (batch_size, length, d_model)
        out = self.prenet(mels)
        out = self.dropout(out)

        # ConformerBlock expects features in the shape of (batch_size, length, d_model)
        out = self.encoder_layer(out)  # (batch_size, length, d_model)

        # Mean pooling
        stats = out.mean(dim=1)  # (batch_size, d_model)

        # out: (batch_size, n_spks)
        out = self.pred_layer(stats)
        return out

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        testdata_path = self.data_dir / "testdata.json"
        with testdata_path.open() as f:
            metadata = json.load(f)
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def parse_args():
    """Parse configuration arguments."""
    config = {
        "data_dir": "./Dataset",
        "model_path": "./model.ckpt",
        "output_path": "./output.csv",
    }
    return config


def main(data_dir, model_path, output_path):
    """Main function for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Using {device}")

    # Load mapping
    mapping_path = Path(data_dir) / "mapping.json"
    with mapping_path.open() as f:
        mapping = json.load(f)

    # Initialize dataset and dataloader
    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print("[Info]: Data loading complete!", flush=True)

    # Initialize model
    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("[Info]: Model creation complete!", flush=True)

    # Perform inference
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == "__main__":
    config = parse_args()
    main(**config)