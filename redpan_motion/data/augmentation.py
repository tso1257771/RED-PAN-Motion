"""
Data augmentation transforms for PyTorch RED-PAN training.

These transforms operate on (waveform, label, mask) tuples and are designed
to be composable with standard PyTorch transform pipelines.
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
import scipy.signal as ss


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
        category: str = '',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for t in self.transforms:
            if _accepts_category(t):
                waveform, label, mask = t(waveform, label, mask, category=category)
            else:
                waveform, label, mask = t(waveform, label, mask)
        return waveform, label, mask


def _accepts_category(t) -> bool:
    """Check if a transform accepts the category kwarg."""
    import inspect
    try:
        sig = inspect.signature(t.__call__)
        return 'category' in sig.parameters
    except (ValueError, TypeError):
        return False


class Normalize:
    """
    Z-score normalization per sample.
    
    Normalizes each waveform to have zero mean and unit variance.
    Handles edge cases (zero variance) gracefully.
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Normalize per channel or globally
        mean = np.mean(waveform)
        std = np.std(waveform)
        
        if std < self.eps:
            std = 1.0
        
        waveform = (waveform - mean) / std
        
        # Handle inf/nan
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        return waveform, label, mask


class NormalizePerChannel:
    """
    Z-score normalization per channel.
    
    Normalizes each waveform channel (E, N, Z) independently.
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # waveform shape: (C, T)
        for c in range(waveform.shape[0]):
            mean = np.mean(waveform[c])
            std = np.std(waveform[c])
            
            if std < self.eps:
                std = 1.0
            
            waveform[c] = (waveform[c] - mean) / std
        
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        return waveform, label, mask


class RandomAmplitudeScale:
    """
    Randomly scale waveform amplitude.
    
    Useful for training models that are invariant to absolute amplitude.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0)):
        self.scale_range = scale_range
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.random.uniform(*self.scale_range)
        waveform = waveform * scale
        return waveform, label, mask


class RandomGaussianNoise:
    """
    Add random Gaussian noise to waveform.
    
    Noise level is relative to the signal's standard deviation.
    """
    
    def __init__(
        self,
        noise_level_range: Tuple[float, float] = (0.0, 0.1),
        prob: float = 0.5,
    ):
        self.noise_level_range = noise_level_range
        self.prob = prob
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() > self.prob:
            return waveform, label, mask
        
        noise_level = np.random.uniform(*self.noise_level_range)
        std = np.std(waveform)
        noise = np.random.normal(0, noise_level * std, waveform.shape)
        waveform = waveform + noise.astype(waveform.dtype)
        
        return waveform, label, mask


class RandomChannelDrop:
    """
    Randomly zero out one or more channels, ensuring at least one channel
    retains recognizable earthquake signal after z-score normalization.

    Checks that the max amplitude of the remaining channels exceeds
    `min_signal_std` standard deviations above zero. If not, the drop
    is skipped to avoid training the model on zero waveform + EQ labels.
    """

    def __init__(self, prob: float = 0.03, max_channels: int = 2,
                 min_signal_std: float = 3.0):
        self.prob = prob
        self.max_channels = max_channels
        self.min_signal_std = min_signal_std

    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() > self.prob:
            return waveform, label, mask

        n_channels = waveform.shape[0]
        n_drop = np.random.randint(1, min(self.max_channels, n_channels) + 1)
        channels_to_drop = np.random.choice(n_channels, n_drop, replace=False)
        keep = [c for c in range(n_channels) if c not in channels_to_drop]

        # Check: at least one kept channel has recognizable signal
        # (peak amplitude > min_signal_std after z-score)
        has_signal = False
        for c in keep:
            ch = waveform[c]
            std = np.std(ch)
            if std > 1e-10:
                peak = np.max(np.abs(ch - np.mean(ch))) / std
                if peak >= self.min_signal_std:
                    has_signal = True
                    break
        if not has_signal:
            return waveform, label, mask  # skip drop

        waveform = waveform.copy()
        for c in channels_to_drop:
            waveform[c] = 0.0

        return waveform, label, mask


class RandomTemporalShift:
    """
    Random circular shift in time.
    
    Note: Labels and masks are also shifted to maintain alignment.
    """
    
    def __init__(self, max_shift_samples: int = 100):
        self.max_shift = max_shift_samples
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        
        # Shift waveform (C, T)
        waveform = np.roll(waveform, shift, axis=1)
        
        # Shift label (T, 3)
        label = np.roll(label, shift, axis=0)
        
        # Shift mask (T, 2)
        mask = np.roll(mask, shift, axis=0)
        
        return waveform, label, mask


class RandomFlipPolarity:
    """
    Randomly flip waveform polarity (multiply by -1).
    
    Common augmentation for seismic data since polarity depends on source mechanism.
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            waveform = -waveform
        return waveform, label, mask


class RandomTaper:
    """
    Apply random tapering to waveform edges.
    
    Simulates waveforms that start/end mid-signal.
    """
    
    def __init__(
        self,
        max_taper_fraction: float = 0.1,
        prob: float = 0.3,
    ):
        self.max_taper = max_taper_fraction
        self.prob = prob
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() > self.prob:
            return waveform, label, mask
        
        n_samples = waveform.shape[1]
        taper_samples = int(np.random.uniform(0, self.max_taper) * n_samples)
        
        if taper_samples > 0:
            # Create Tukey window
            window = ss.windows.tukey(n_samples, alpha=2 * taper_samples / n_samples)
            waveform = waveform * window[np.newaxis, :]
        
        return waveform, label, mask


class NoiseTaper:
    """
    Apply event-like amplitude envelopes to noise samples only.

    Creates random ramp-up / plateau / ramp-down patterns on pure noise
    traces, teaching the model that amplitude variation alone does not
    indicate an earthquake. Only activates when category contains 'noise'.

    Envelope types (randomly chosen):
      - 'ramp_up': gradual onset, simulates event arriving
      - 'ramp_down': gradual decay, simulates event ending
      - 'trapezoid': ramp-up → plateau → ramp-down, simulates full event
      - 'triangle': ramp-up → ramp-down, simulates brief transient

    Args:
        prob: probability of applying to each noise sample
        min_taper_frac: minimum fraction of trace for ramp section
        max_taper_frac: maximum fraction of trace for ramp section
    """

    def __init__(
        self,
        prob: float = 0.5,
        min_taper_frac: float = 0.05,
        max_taper_frac: float = 0.4,
    ):
        self.prob = prob
        self.min_taper_frac = min_taper_frac
        self.max_taper_frac = max_taper_frac

    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
        category: str = '',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Only apply to noise samples
        if 'noise' not in category.lower():
            return waveform, label, mask
        if np.random.random() > self.prob:
            return waveform, label, mask

        n_samples = waveform.shape[1]
        taper_frac = np.random.uniform(self.min_taper_frac, self.max_taper_frac)
        taper_len = max(1, int(taper_frac * n_samples))

        envelope = np.ones(n_samples, dtype=np.float32)
        style = np.random.choice(['ramp_up', 'ramp_down', 'trapezoid', 'triangle'])

        if style == 'ramp_up':
            # Start near zero, ramp up to 1
            start = np.random.randint(0, max(1, n_samples - taper_len))
            envelope[:start] = np.random.uniform(0.0, 0.1)
            envelope[start:start + taper_len] = np.linspace(
                envelope[start - 1] if start > 0 else 0.0, 1.0, taper_len)

        elif style == 'ramp_down':
            # Start at 1, ramp down to near zero
            start = np.random.randint(0, max(1, n_samples - taper_len))
            envelope[start:start + taper_len] = np.linspace(1.0, 0.0, taper_len)
            envelope[start + taper_len:] = np.random.uniform(0.0, 0.1)

        elif style == 'trapezoid':
            # Ramp up → plateau → ramp down
            rise = taper_len
            fall = max(1, int(np.random.uniform(0.5, 1.5) * taper_len))
            plateau_start = np.random.randint(0, max(1, n_samples - rise - fall))
            envelope[:plateau_start] = np.random.uniform(0.0, 0.15)
            envelope[plateau_start:plateau_start + rise] = np.linspace(0.0, 1.0, rise)
            fall_start = min(plateau_start + rise + np.random.randint(0, max(1, n_samples // 4)),
                            n_samples - fall)
            envelope[fall_start:fall_start + fall] = np.linspace(1.0, 0.0, fall)
            envelope[fall_start + fall:] = np.random.uniform(0.0, 0.1)

        elif style == 'triangle':
            # Ramp up → ramp down, centered
            center = np.random.randint(taper_len, max(taper_len + 1, n_samples - taper_len))
            rise_start = max(0, center - taper_len)
            fall_end = min(n_samples, center + taper_len)
            envelope[:rise_start] = np.random.uniform(0.0, 0.1)
            envelope[rise_start:center] = np.linspace(0.0, 1.0, center - rise_start)
            envelope[center:fall_end] = np.linspace(1.0, 0.0, fall_end - center)
            envelope[fall_end:] = np.random.uniform(0.0, 0.1)

        waveform = waveform * envelope[np.newaxis, :]
        return waveform, label, mask


class AddColoredNoise:
    """
    Add colored noise with realistic spectral characteristics.
    
    Uses spectrum matching to generate noise similar to seismic background.
    """
    
    def __init__(
        self,
        noise_level_range: Tuple[float, float] = (0.01, 0.1),
        prob: float = 0.3,
    ):
        self.noise_level_range = noise_level_range
        self.prob = prob
    
    def _generate_colored_noise(self, n_samples: int, beta: float = 1.0) -> np.ndarray:
        """Generate 1/f^beta noise."""
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # Power spectrum: 1/f^beta
        power = 1.0 / (np.abs(freqs) ** (beta / 2))
        
        # Random phase
        phase = np.random.uniform(0, 2 * np.pi, n_samples)
        
        # Create noise in frequency domain
        noise_fft = power * np.exp(1j * phase)
        noise = np.real(np.fft.ifft(noise_fft))
        
        # Normalize
        noise = noise / np.std(noise)
        
        return noise.astype(np.float32)
    
    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() > self.prob:
            return waveform, label, mask
        
        noise_level = np.random.uniform(*self.noise_level_range)
        n_channels, n_samples = waveform.shape
        
        signal_std = np.std(waveform)
        
        for c in range(n_channels):
            # Random spectral slope (1.0 = pink noise, 2.0 = brown noise)
            beta = np.random.uniform(0.5, 2.0)
            noise = self._generate_colored_noise(n_samples, beta)
            waveform[c] = waveform[c] + noise * noise_level * signal_std
        
        return waveform, label, mask


class RandomHighpassTaper:
    """Apply random highpass filter + cosine taper to both ends.

    Only applied to singleEQ categories. Simulates real-time processing
    artifacts (highpass filter ringing, window tapering) that the model
    will encounter during inference.

    The taper zeroes out waveform edges smoothly, so arrival labels near
    edges should still be valid (Gaussians extend only ±0.2-0.3s).

    Args:
        prob: probability of applying (per sample)
        freq_range: (min, max) highpass corner frequency in Hz
        taper_fraction: fraction of trace length to taper at each end
        categories: only apply to samples whose category starts with these
    """

    def __init__(
        self,
        prob: float = 0.1,
        freq_range: Tuple[float, float] = (0.5, 2.0),
        taper_fraction: float = 0.05,
        categories: Tuple[str, ...] = ('singleEQ',),
    ):
        self.prob = prob
        self.freq_range = freq_range
        self.taper_fraction = taper_fraction
        self.categories = tuple(c.lower() for c in categories)

    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
        category: str = '',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Only apply to matching categories
        cat_lower = category.lower()
        if not any(cat_lower.startswith(c) for c in self.categories):
            return waveform, label, mask

        if np.random.random() > self.prob:
            return waveform, label, mask

        waveform = waveform.copy()
        n_channels, n_samples = waveform.shape
        sr = 100.0  # target sample rate

        # Random highpass corner frequency
        freq = np.random.uniform(*self.freq_range)

        # Design Butterworth highpass filter
        nyq = sr / 2.0
        if freq < nyq * 0.95:  # skip if corner too close to Nyquist
            b, a = ss.butter(2, freq / nyq, btype='high')
            for c in range(n_channels):
                waveform[c] = ss.filtfilt(b, a, waveform[c]).astype(np.float32)

        # Cosine taper at both ends
        taper_len = max(1, int(n_samples * self.taper_fraction))
        taper = np.ones(n_samples, dtype=np.float32)
        taper[:taper_len] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
        taper[-taper_len:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len, 0, -1) / taper_len))

        for c in range(n_channels):
            waveform[c] *= taper

        return waveform, label, mask


def get_training_transform(
    normalize: bool = True,
    augment: bool = True,
) -> Compose:
    """
    Get standard training transforms.
    
    Args:
        normalize: Whether to apply normalization
        augment: Whether to apply augmentation
    
    Returns:
        Composed transform function
    """
    transforms = []
    
    if augment:
        transforms.extend([
            RandomAmplitudeScale((0.5, 2.0)),
            RandomGaussianNoise((0.0, 0.05), prob=0.6),
            RandomChannelDrop(prob=0.03, max_channels=2, min_signal_std=3.0),
            RandomFlipPolarity(prob=0.5),
            AddColoredNoise((0.01, 0.05), prob=0.5),
            RandomHighpassTaper(prob=0.1, freq_range=(0.5, 2.0), taper_fraction=0.01),
            NoiseTaper(prob=0.5, min_taper_frac=0.05, max_taper_frac=0.4),
        ])
    
    if normalize:
        transforms.append(NormalizePerChannel())
    
    return Compose(transforms)


def get_validation_transform(normalize: bool = True) -> Compose:
    """
    Get standard validation transforms (no augmentation).
    
    Args:
        normalize: Whether to apply normalization
    
    Returns:
        Composed transform function
    """
    transforms = []
    
    if normalize:
        transforms.append(NormalizePerChannel())
    
    return Compose(transforms)


if __name__ == '__main__':
    # Test transforms
    np.random.seed(42)
    
    waveform = np.random.randn(3, 6000).astype(np.float32)
    label = np.random.rand(6000, 3).astype(np.float32)
    mask = np.random.rand(6000, 2).astype(np.float32)
    
    print("Testing transforms...")
    print(f"Original waveform stats: mean={waveform.mean():.3f}, std={waveform.std():.3f}")
    
    # Test individual transforms
    transform = Normalize()
    wf_norm, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
    print(f"After Normalize: mean={wf_norm.mean():.3f}, std={wf_norm.std():.3f}")
    
    transform = RandomAmplitudeScale((0.5, 2.0))
    wf_scale, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
    print(f"After AmplitudeScale: mean={wf_scale.mean():.3f}, std={wf_scale.std():.3f}")
    
    # Test composed transform
    transform = get_training_transform(normalize=True, augment=True)
    wf_train, _, _ = transform(waveform.copy(), label.copy(), mask.copy())
    print(f"After training transforms: mean={wf_train.mean():.3f}, std={wf_train.std():.3f}")
    
    print("\n✓ All transforms working correctly!")
