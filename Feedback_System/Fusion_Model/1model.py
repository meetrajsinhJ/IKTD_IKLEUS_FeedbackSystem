import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors"""
    batch_size = len(batch)

    # Get maximum sequence length for text features in this batch
    max_seq_length = max(b['text_features'].size(0) for b in batch)

    # Initialize tensors for the batch
    visual_features = torch.stack([b['visual_features'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])

    # Pad text features to max_seq_length
    padded_text_features = torch.zeros(batch_size, max_seq_length, batch[0]['text_features'].size(-1))
    attention_mask = torch.zeros(batch_size, max_seq_length, dtype=torch.bool)

    for idx, b in enumerate(batch):
        cur_seq_len = b['text_features'].size(0)
        padded_text_features[idx, :cur_seq_len] = b['text_features']
        attention_mask[idx, :cur_seq_len] = 1

    return {
        'visual_features': visual_features,
        'text_features': padded_text_features,
        'attention_mask': attention_mask,
        'labels': labels,
        'sample_ids': [b['sample_id'] for b in batch]
    }


class TechnicalDrawingDataset(Dataset):
    """Dataset class for processed technical drawing samples"""

    def __init__(self, data_dir: Path, max_annotations: int = 10):
        self.data_dir = Path(data_dir)
        self.sample_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]
        self.max_annotations = max_annotations

        print(f"Found {len(self.sample_ids)} samples in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        try:
            sample_dir = self.data_dir / self.sample_ids[idx]

            # Load features
            incorrect_features = np.load(sample_dir / 'incorrect_features.npy', allow_pickle=True).item()
            correct_features = np.load(sample_dir / 'correct_features.npy', allow_pickle=True).item()
            annotation_features = np.load(sample_dir / 'annotation_features.npy', allow_pickle=True)
            labels = np.load(sample_dir / 'labels.npy', allow_pickle=True).item()

            # Process features
            visual_features = self._process_visual_features(incorrect_features, correct_features)
            text_features = self._process_text_features(annotation_features)
            label_tensor = self._process_labels(labels)

            return {
                'visual_features': visual_features,
                'text_features': text_features,
                'labels': label_tensor,
                'sample_id': self.sample_ids[idx]
            }

        except Exception as e:
            print(f"Error loading sample {self.sample_ids[idx]}: {str(e)}")
            # Return zero tensors as fallback
            return {
                'visual_features': torch.zeros(256),
                'text_features': torch.zeros(self.max_annotations, 384),
                'labels': torch.zeros(9),
                'sample_id': self.sample_ids[idx]
            }

    def _process_visual_features(self, incorrect: Dict, correct: Dict) -> torch.Tensor:
        """Process visual features from both drawings"""
        try:
            # Extract features from incorrect drawing
            inc_features = []
            inc_features.extend(self._flatten_dict_values(incorrect['visual_features']))
            inc_features.extend([
                float(incorrect['structural_features']['lines']),
                float(incorrect['structural_features']['contours'])
            ])

            # Extract features from correct drawing
            cor_features = []
            cor_features.extend(self._flatten_dict_values(correct['visual_features']))
            cor_features.extend([
                float(correct['structural_features']['lines']),
                float(correct['structural_features']['contours'])
            ])

            # Combine and pad/truncate to fixed size
            all_features = inc_features + cor_features
            target_size = 256  # Adjust based on your needs

            if len(all_features) > target_size:
                all_features = all_features[:target_size]
            else:
                all_features.extend([0] * (target_size - len(all_features)))

            return torch.FloatTensor(all_features)

        except Exception as e:
            print(f"Error processing visual features: {str(e)}")
            return torch.zeros(256)

    def _flatten_dict_values(self, d: Dict) -> List[float]:
        """Recursively flatten dictionary values"""
        flattened = []
        for v in d.values():
            if isinstance(v, dict):
                flattened.extend(self._flatten_dict_values(v))
            elif isinstance(v, (list, np.ndarray)):
                if isinstance(v, np.ndarray) and v.dtype == np.dtype('object'):
                    continue
                flattened.extend([float(x) for x in v])
            elif isinstance(v, (int, float)):
                flattened.append(float(v))
        return flattened

    def _process_text_features(self, annotations: List[Dict]) -> torch.Tensor:
        """Process text features with fixed size output"""
        try:
            processed_features = []

            for ann in annotations[:self.max_annotations]:
                features = []

                # Add embedding if available
                if 'embedding' in ann and ann['embedding'] is not None:
                    if isinstance(ann['embedding'], list):
                        features.extend(ann['embedding'])
                    elif isinstance(ann['embedding'], np.ndarray):
                        features.extend(ann['embedding'].tolist())

                # Add other features
                if features:
                    # Pad or truncate to fixed size
                    if len(features) > 384:
                        features = features[:384]
                    else:
                        features.extend([0] * (384 - len(features)))

                    processed_features.append(features)

            # Pad to max_annotations
            while len(processed_features) < self.max_annotations:
                processed_features.append([0] * 384)

            return torch.FloatTensor(processed_features)

        except Exception as e:
            print(f"Error processing text features: {str(e)}")
            return torch.zeros((self.max_annotations, 384))

    def _process_labels(self, labels: Dict) -> torch.Tensor:
        """Process labels into binary tensor"""
        try:
            label_values = []
            for criterion_id in range(1, 10):  # 9 criteria
                status = labels['verification_status'].get(criterion_id, False)
                label_values.append(float(status))

            return torch.FloatTensor(label_values)

        except Exception as e:
            print(f"Error processing labels: {str(e)}")
            return torch.zeros(9)

class MultiFusionModel(nn.Module):
    def __init__(self,
                 visual_dim: int = 256,
                 text_dim: int = 384,
                 hidden_dim: int = 256,
                 num_criteria: int = 9,
                 max_seq_length: int = 10):
        super().__init__()

        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Multi-head attention for fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Positional encoding for sequence
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_length, hidden_dim))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_criteria),
            nn.Sigmoid()
        )

    def forward(self, visual_features, text_features, attention_mask=None):
        batch_size = visual_features.size(0)

        # Encode features
        visual_encoded = self.visual_encoder(visual_features)
        text_encoded = self.text_encoder(text_features)

        # Add positional encoding to text features
        text_encoded = text_encoded + self.pos_encoder[:, :text_encoded.size(1)]

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            text_encoded = text_encoded.masked_fill(~attention_mask.unsqueeze(-1), 0)

        # Fusion through attention
        visual_query = visual_encoded.unsqueeze(1)
        fused_features, _ = self.fusion_attention(
            visual_query,
            text_encoded,
            text_encoded,
            key_padding_mask=None if attention_mask is None else ~attention_mask
        )

        # Combine features
        combined = torch.cat([
            visual_encoded,
            fused_features.squeeze(1)
        ], dim=-1)

        # Generate predictions
        return self.output_layer(combined)

def train_model(data_dir: Path,
                model: nn.Module,
                num_epochs: int = 5,
                batch_size: int = 4,
                learning_rate: float = 1e-4,
                checkpoint_dir: Path = Path("checkpoints")):
    """Train the multifusion model with checkpoints and model saving"""

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    # Create dataset and dataloader
    dataset = TechnicalDrawingDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize best loss for model saving
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                visual_features = batch['visual_features'].to(device)
                text_features = batch['text_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                predictions = model(visual_features, text_features, attention_mask)
                loss = criterion(predictions, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

                # Save periodic checkpoint (every 100 batches)
                if (batch_idx + 1) % 100 == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}_batch_{batch_idx + 1}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'epoch': epoch,
                            'batch': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }
                    )

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')

        # Save epoch checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(
            checkpoint_path,
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
        )

        # Save best model if current loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                best_model_path,
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }
            )
            print(f"Saved best model with loss: {best_loss:.4f}")

    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(
        final_model_path,
        {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
    )
    print(f"Saved final model at: {final_model_path}")

    return model


def save_checkpoint(path: Path, state_dict: Dict):
    """Save model checkpoint with error handling"""
    try:
        torch.save(state_dict, path)
        print(f"Successfully saved checkpoint to {path}")
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {str(e)}")


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
    """Load model checkpoint with error handling"""
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint from {path}: {str(e)}")
        return None


def main():
    # Setup paths
    data_dir = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPipeline/processed_data")
    checkpoint_dir = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/Multifusion/checkpoints")

    # Initialize model
    model = MultiFusionModel(
        visual_dim=256,
        text_dim=384,
        hidden_dim=256,
        num_criteria=9,
        max_seq_length=10
    )

    # Training parameters
    train_params = {
        'num_epochs': 5,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'checkpoint_dir': checkpoint_dir
    }

    try:
        # Train model
        trained_model = train_model(
            data_dir=data_dir,
            model=model,
            **train_params
        )

        print("Training completed successfully!")
        print(f"Model checkpoints saved in: {checkpoint_dir}")
        print(f"Best model saved as: {checkpoint_dir / 'best_model.pt'}")
        print(f"Final model saved as: {checkpoint_dir / 'final_model.pt'}")

        # Example: Load best model
        best_model = MultiFusionModel(
            visual_dim=256,
            text_dim=384,
            hidden_dim=256,
            num_criteria=9,
            max_seq_length=10
        )
        checkpoint = load_checkpoint(checkpoint_dir / 'best_model.pt', best_model)
        if checkpoint:
            print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
