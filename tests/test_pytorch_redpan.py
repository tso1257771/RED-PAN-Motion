"""
Unit tests for PyTorch RED-PAN implementation.

Run with: pytest tests/test_pytorch_redpan.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import h5py


class TestMTANR2UNet:
    """Tests for the MTAN R2U-Net model."""
    
    @pytest.fixture
    def model(self):
        from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
        torch.manual_seed(1)        # deterministic weights -> reproducible tests
        return MTAN_R2UNet()
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_model_creation(self, model):
        """Test model can be instantiated."""
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_parameter_count(self, model):
        """Test model has expected number of parameters."""
        n_params = model.count_parameters()
        # Base MTAN_R2UNet at the default config is ~0.5M parameters.
        assert 100_000 < n_params < 600_000
    
    def test_forward_pass_cpu(self, model):
        """Test forward pass on CPU."""
        batch_size = 2
        seq_len = 6000
        x = torch.randn(batch_size, 3, seq_len)
        
        model.eval()
        with torch.no_grad():
            picker, detector = model(x)
        
        assert picker.shape == (batch_size, 3, seq_len)
        assert detector.shape == (batch_size, 2, seq_len)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_gpu(self, model, device):
        """Test forward pass on GPU."""
        model = model.to(device)
        x = torch.randn(2, 3, 6000).to(device)
        
        model.eval()
        with torch.no_grad():
            picker, detector = model(x)
        
        assert picker.device.type == 'cuda'
        assert detector.device.type == 'cuda'
    
    def test_output_probabilities(self, model):
        """Test outputs are valid probabilities (sum to 1)."""
        x = torch.randn(2, 3, 6000)
        
        model.eval()
        with torch.no_grad():
            picker, detector = model(x)
        
        # Check probabilities sum to 1
        picker_sum = picker.sum(dim=1)  # Sum over classes
        detector_sum = detector.sum(dim=1)
        
        assert torch.allclose(picker_sum, torch.ones_like(picker_sum), atol=1e-5)
        assert torch.allclose(detector_sum, torch.ones_like(detector_sum), atol=1e-5)
    
    def test_gradient_flow(self, model):
        """Test gradients flow through the entire network."""
        x = torch.randn(2, 3, 6000, requires_grad=True)
        
        model.train()
        picker, detector = model(x)
        loss = picker.sum() + detector.sum()
        loss.backward()
        
        # Check input received gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_different_sequence_lengths(self, model):
        """Test model handles various sequence lengths."""
        model.eval()
        
        for seq_len in [1000, 3000, 6000, 9000]:
            x = torch.randn(1, 3, seq_len)
            with torch.no_grad():
                picker, detector = model(x)
            
            assert picker.shape[2] == seq_len
            assert detector.shape[2] == seq_len
    
    def test_batch_size_invariance(self, model):
        """Test model produces consistent outputs regardless of batch size."""
        model.eval()
        torch.manual_seed(42)
        
        # Single sample
        x1 = torch.randn(1, 3, 6000)
        
        # Same sample in batch
        x_batch = x1.repeat(4, 1, 1)
        
        with torch.no_grad():
            picker1, detector1 = model(x1)
            picker_batch, detector_batch = model(x_batch)
        
        # Identical inputs across the batch must yield identical outputs (samples
        # are processed independently). atol=1e-4 absorbs CPU conv float-rounding
        # that softmax can amplify on untrained weights; a real batch-mixing bug
        # diverges by O(0.1+). Model weights are seeded in the `model` fixture.
        for i in range(4):
            assert torch.allclose(picker1[0], picker_batch[i], atol=1e-4)


class TestDataset:
    """Tests for data loading utilities."""
    
    @pytest.fixture
    def temp_hdf5_dir(self):
        """Create temporary directory with test HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 50
            h5_path = os.path.join(tmpdir, 'test.h5')
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('waveforms', 
                                data=np.random.randn(n_samples, 3, 6000).astype(np.float32))
                f.create_dataset('labels',
                                data=np.random.rand(n_samples, 6000, 3).astype(np.float32))
                f.create_dataset('masks',
                                data=np.random.rand(n_samples, 6000, 2).astype(np.float32))
            
            yield tmpdir
    
    def test_hdf5_dataset_creation(self, temp_hdf5_dir):
        """Test HDF5 dataset can be created."""
        from redpan_motion.data.dataset import HDF5SeismicDataset
        
        dataset = HDF5SeismicDataset(temp_hdf5_dir)
        assert len(dataset) == 50
    
    def test_hdf5_dataset_getitem(self, temp_hdf5_dir):
        """Test dataset __getitem__ returns correct shapes."""
        from redpan_motion.data.dataset import HDF5SeismicDataset
        
        dataset = HDF5SeismicDataset(temp_hdf5_dir)
        sample = dataset[0]
        
        assert 'waveform' in sample
        assert 'label' in sample
        assert 'mask' in sample
        
        assert sample['waveform'].shape == (3, 6000)
        assert sample['label'].shape == (6000, 3)
        assert sample['mask'].shape == (6000, 2)
    
    def test_dataloader_creation(self, temp_hdf5_dir):
        """Test DataLoader works with dataset."""
        from redpan_motion.data.dataset import HDF5SeismicDataset, create_dataloader
        
        dataset = HDF5SeismicDataset(temp_hdf5_dir)
        loader = create_dataloader(dataset, batch_size=8, num_workers=0)
        
        batch = next(iter(loader))
        
        assert batch['waveform'].shape[0] == 8
        assert batch['waveform'].shape[1] == 3
        assert batch['waveform'].shape[2] == 6000
    
    def test_numpy_dataset(self):
        """Test NumpySeismicDataset."""
        from redpan_motion.data.dataset import NumpySeismicDataset
        
        n_samples = 20
        waveforms = np.random.randn(n_samples, 3, 6000).astype(np.float32)
        labels = np.random.rand(n_samples, 6000, 3).astype(np.float32)
        masks = np.random.rand(n_samples, 6000, 2).astype(np.float32)
        
        dataset = NumpySeismicDataset(waveforms, labels, masks)
        
        assert len(dataset) == n_samples
        
        sample = dataset[0]
        assert sample['waveform'].shape == (3, 6000)


class TestAugmentation:
    """Tests for data augmentation transforms."""
    
    @pytest.fixture
    def sample_data(self):
        waveform = np.random.randn(3, 6000).astype(np.float32)
        label = np.random.rand(6000, 3).astype(np.float32)
        mask = np.random.rand(6000, 2).astype(np.float32)
        return waveform, label, mask
    
    def test_normalize(self, sample_data):
        """Test Normalize transform."""
        from redpan_motion.data.augmentation import Normalize
        
        waveform, label, mask = sample_data
        transform = Normalize()
        
        wf_out, label_out, mask_out = transform(waveform.copy(), label.copy(), mask.copy())
        
        # Check normalized
        assert np.abs(wf_out.mean()) < 0.1
        assert np.abs(wf_out.std() - 1.0) < 0.1
        
        # Labels and masks unchanged
        np.testing.assert_array_equal(label_out, label)
        np.testing.assert_array_equal(mask_out, mask)
    
    def test_random_amplitude_scale(self, sample_data):
        """Test RandomAmplitudeScale transform."""
        from redpan_motion.data.augmentation import RandomAmplitudeScale
        
        waveform, label, mask = sample_data
        transform = RandomAmplitudeScale(scale_range=(0.5, 2.0))
        
        np.random.seed(42)
        wf_out, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
        
        # Check scaling happened
        ratio = np.std(wf_out) / np.std(waveform)
        assert 0.5 <= ratio <= 2.0
    
    def test_random_flip_polarity(self, sample_data):
        """Test RandomFlipPolarity transform."""
        from redpan_motion.data.augmentation import RandomFlipPolarity
        
        waveform, label, mask = sample_data
        transform = RandomFlipPolarity(prob=1.0)  # Always flip
        
        wf_out, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
        
        np.testing.assert_array_almost_equal(wf_out, -waveform)
    
    def test_compose_transforms(self, sample_data):
        """Test Compose chains transforms correctly."""
        from redpan_motion.data.augmentation import Compose, Normalize, RandomFlipPolarity
        
        waveform, label, mask = sample_data
        
        transform = Compose([
            RandomFlipPolarity(prob=1.0),
            Normalize(),
        ])
        
        wf_out, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
        
        # Should be flipped and normalized
        assert np.abs(wf_out.mean()) < 0.1
    
    def test_training_transform(self, sample_data):
        """Test get_training_transform creates valid transform."""
        from redpan_motion.data.augmentation import get_training_transform
        
        waveform, label, mask = sample_data
        transform = get_training_transform()
        
        # Should not raise
        wf_out, label_out, mask_out = transform(waveform.copy(), label.copy(), mask.copy())
        
        assert wf_out.shape == waveform.shape


class TestLosses:
    """Tests for loss functions."""
    
    @pytest.fixture
    def predictions(self):
        batch_size = 4
        seq_len = 6000
        
        picker_pred = torch.softmax(torch.randn(batch_size, 3, seq_len), dim=1)
        detector_pred = torch.softmax(torch.randn(batch_size, 2, seq_len), dim=1)
        
        picker_target = torch.zeros(batch_size, seq_len, 3)
        picker_target[..., 2] = 1.0  # All noise
        
        detector_target = torch.zeros(batch_size, seq_len, 2)
        detector_target[..., 1] = 1.0  # All unmask
        
        return picker_pred, detector_pred, picker_target, detector_target
    
    def test_categorical_crossentropy(self, predictions):
        """Test CategoricalCrossEntropy loss."""
        from redpan_motion.training.losses import CategoricalCrossEntropy
        
        picker_pred, _, picker_target, _ = predictions
        
        loss_fn = CategoricalCrossEntropy()
        loss = loss_fn(picker_pred, picker_target)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_multitask_loss(self, predictions):
        """Test MultiTaskLoss."""
        from redpan_motion.training.losses import MultiTaskLoss
        
        picker_pred, detector_pred, picker_target, detector_target = predictions
        
        loss_fn = MultiTaskLoss(use_dwa=True)
        combined, picker_loss, detector_loss = loss_fn(
            picker_pred, detector_pred,
            picker_target, detector_target,
            return_individual=True
        )
        
        assert combined.item() > 0
        assert picker_loss.item() > 0
        assert detector_loss.item() > 0
    
    def test_dwa_weight_update(self, predictions):
        """Test DWA weight updates over epochs."""
        from redpan_motion.training.losses import MultiTaskLoss
        
        loss_fn = MultiTaskLoss(use_dwa=True)
        
        initial_weights = loss_fn.get_weights().clone()
        
        # Simulate epochs with changing losses
        loss_fn.update_weights(torch.tensor([1.0, 1.0]))
        loss_fn.update_weights(torch.tensor([0.8, 1.2]))
        
        updated_weights = loss_fn.get_weights()
        
        # Weights should have changed
        assert not torch.allclose(initial_weights, updated_weights)


class TestPredictor:
    """Tests for inference predictor."""
    
    @pytest.fixture
    def predictor(self):
        from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
        from redpan_motion.inference.predictor import REDPANPredictor
        
        model = MTAN_R2UNet()
        return REDPANPredictor(model=model, batch_size=4)
    
    def test_predict_short_waveform(self, predictor):
        """Test prediction on waveform shorter than model input."""
        waveform = np.random.randn(3, 3000).astype(np.float32)
        
        picker, detector = predictor.predict_array(waveform)
        
        assert picker.shape == (3000, 3)
        assert detector.shape == (3000, 2)
    
    def test_predict_exact_length(self, predictor):
        """Test prediction on waveform exactly matching model input."""
        waveform = np.random.randn(3, 6000).astype(np.float32)
        
        picker, detector = predictor.predict_array(waveform)
        
        assert picker.shape == (6000, 3)
        assert detector.shape == (6000, 2)
    
    def test_predict_long_waveform(self, predictor):
        """Test sliding window prediction on long waveform."""
        waveform = np.random.randn(3, 30000).astype(np.float32)
        
        picker, detector = predictor.predict_array(waveform)
        
        assert picker.shape == (30000, 3)
        assert detector.shape == (30000, 2)
    
    def test_output_valid_probabilities(self, predictor):
        """Test predictor outputs valid probabilities."""
        waveform = np.random.randn(3, 6000).astype(np.float32)
        
        picker, detector = predictor.predict_array(waveform)
        
        # Check probabilities sum to ~1
        picker_sum = picker.sum(axis=1)
        detector_sum = detector.sum(axis=1)
        
        np.testing.assert_array_almost_equal(picker_sum, np.ones_like(picker_sum), decimal=4)
        np.testing.assert_array_almost_equal(detector_sum, np.ones_like(detector_sum), decimal=4)


class TestTrainer:
    """Tests for training utilities."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from redpan_motion.training.trainer import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
    
    def test_trainer_creation(self):
        """Test REDPANTrainer can be instantiated."""
        from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
        from redpan_motion.data.dataset import NumpySeismicDataset, create_dataloader
        from redpan_motion.training.trainer import REDPANTrainer, TrainingConfig
        
        model = MTAN_R2UNet()
        
        # Minimal dataset
        waveforms = np.random.randn(16, 3, 6000).astype(np.float32)
        labels = np.zeros((16, 6000, 3), dtype=np.float32)
        labels[..., 2] = 1.0
        masks = np.zeros((16, 6000, 2), dtype=np.float32)
        masks[..., 1] = 1.0
        
        dataset = NumpySeismicDataset(waveforms, labels, masks)
        loader = create_dataloader(dataset, batch_size=4, num_workers=0)
        
        config = TrainingConfig(epochs=1, checkpoint_dir='/tmp/test_ckpt')
        
        trainer = REDPANTrainer(model=model, train_loader=loader, config=config)
        
        assert trainer is not None
        assert trainer.model is model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
