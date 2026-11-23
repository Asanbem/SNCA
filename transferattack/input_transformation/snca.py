import numpy as np
import torch
import torch.nn.functional as F

from ..attack import Attack
from ..utils import *


class SNCA(Attack):
    """
    SNCA (Spatial-Neighborhood Composite Attack)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        copies (int): the number of sampling copies per iteration, each copy
                      contains branch1 (block transform), branch2 (neighborhood
                      noise) and branch3 (noise + transform).
        num_blocks (int): number of blocks along each dimension for the dynamic
                          block transformation.
        flip_prob (float): probability of horizontal flipping for each block.
        max_translate_ratio (float): maximum translation ratio relative to image
                                     dimensions.
        noise_radius (float): scale factor of the neighborhood noise w.r.t. epsilon.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, 'linfty' or 'l2'.
        loss (str): the loss function.
        device (torch.device): the device for computation.

    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='SNCA', copies=20, num_blocks=2, flip_prob=0.5, 
                max_translate_ratio=0.1, noise_radius=0.25, **kwargs):

        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        # Hyperparameters not registered by the base class
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

        # SNCA-specific hyperparameters
        self.copies = copies                            # number of copies K per iteration
        self.num_blocks = num_blocks                    # number of blocks along each dimension
        self.flip_prob = flip_prob                      # probability of horizontal block flipping
        self.max_translate_ratio = max_translate_ratio  # maximum translation ratio relative to image size
        self.noise_radius = noise_radius                # neighborhood noise radius (relative to epsilon)

    # ===================== Spatial Geometry Transformation Components =====================
    def get_block_lengths(self, total_length, num_blocks):
        """Randomly split `total_length` (H or W) into `num_blocks` positive integer block lengths."""
        if num_blocks >= total_length:
            return [1] * total_length
        min_block_size = 1
        rand = np.random.uniform(size=num_blocks)
        rand_norm = np.floor(rand * total_length / rand.sum()).astype(np.int32)
        rand_norm = np.maximum(rand_norm, min_block_size)
        diff = total_length - rand_norm.sum()
        if diff > 0:
            rand_norm[rand_norm.argmax()] += diff
        elif diff < 0:
            idx = rand_norm.argmax()
            rand_norm[idx] = max(min_block_size, rand_norm[idx] + diff)
            while rand_norm.sum() != total_length:
                diff = total_length - rand_norm.sum()
                rand_norm[rand_norm.argmax()] += diff
        return rand_norm.tolist()

    def apply_affine_transform(self, x, theta):
        """Apply batched 2D affine transforms (translation) via affine_grid/grid_sample."""
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def translate_block(self, x, max_translate_ratio=None):
        """Randomly translate each image in the batch within the given ratio of H/W."""
        if max_translate_ratio is None:
            max_translate_ratio = self.max_translate_ratio
        batch_size, _, H, W = x.shape
        if H < 2 or W < 2:
            return x
        device = x.device
        max_tx, max_ty = int(H * max_translate_ratio), int(W * max_translate_ratio)
        tx = torch.randint(-max_tx, max_tx + 1, (batch_size,), device=device).float() / W * 2
        ty = torch.randint(-max_ty, max_ty + 1, (batch_size,), device=device).float() / H * 2
        ones, zeros = torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)
        row1 = torch.stack([ones, zeros, tx], dim=1)
        row2 = torch.stack([zeros, ones, ty], dim=1)
        return self.apply_affine_transform(x, torch.stack([row1, row2], dim=1))

    def flip_block(self, x, flip_prob=None):
        """Randomly flip each image in the batch horizontally with probability `flip_prob`."""
        if flip_prob is None:
            flip_prob = self.flip_prob
        if flip_prob == 0:
            return x
        mask = (torch.rand(x.shape[0], device=x.device) < flip_prob).view(-1, 1, 1, 1)
        return torch.where(mask, torch.flip(x, [3]), x)

    def transform_single_block(self, block):
        """Transform one block: horizontal flip (prob = flip_prob) + random translation."""
        _, _, H, W = block.shape
        if H <= 1 or W <= 1:
            return block
        if self.flip_prob > 0:
            block = self.flip_block(block, self.flip_prob)
        if self.max_translate_ratio > 0:
            block = self.translate_block(block, self.max_translate_ratio)
        return block

    def dynamic_block_transform(self, x):
        """Split the image into random blocks (num_blocks per dimension) and transform each block."""
        _, _, H, W = x.shape
        num_blocks = self.num_blocks
        h_lengths = self.get_block_lengths(H, min(num_blocks, H))
        w_lengths = self.get_block_lengths(W, min(num_blocks, W))
        h_blocks = torch.split(x, h_lengths, dim=2)
        output_rows = []
        for h_block in h_blocks:
            w_blocks = torch.split(h_block, w_lengths, dim=3)
            processed_w = [self.transform_single_block(wb) for wb in w_blocks]
            output_rows.append(torch.cat(processed_w, dim=3))
        return torch.cat(output_rows, dim=2)

    def _normalize_gradient(self, g, eps=1e-12):
        """Scale the gradient by its L1 mean over (C, H, W) for each sample."""
        return g / (torch.mean(torch.abs(g), dim=(1, 2, 3), keepdim=True) + eps)

    def get_logits(self, x, **kwargs):
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits

    # ===================== Attack Loop =====================
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure of SNCA.

        Arguments:
            data (N, C, H, W): tensor for input images
            label (N,): tensor for ground-truth labels if untargeted
            label (2, N): tensor for [ground-truth, targeted labels] if targeted

        Returns:
            delta (N, C, H, W): adversarial perturbations tensor
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device).view(-1)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            grad_trans_sum = torch.zeros_like(data)
            grad_noise_sum = torch.zeros_like(data)
            grad_hyb_sum = torch.zeros_like(data)

            for i in range(self.copies):
                # ========== Branch1: dynamic block transform (flip + translate) ==========
                x_trans = self.dynamic_block_transform(data + delta)
                logits = self.get_logits(x_trans)
                loss = self.get_loss(logits, label)
                g_t = self.get_grad(loss, delta)
                grad_trans_sum += self._normalize_gradient(g_t)

                # ========== Branch2: neighborhood noise ==========
                noise = torch.empty_like(data).uniform_(
                    -self.noise_radius*(i+1) * self.epsilon,
                    self.noise_radius*(i+1) * self.epsilon,
                )
                x_noisy = data + delta + noise
                logits = self.get_logits(x_noisy)
                loss = self.get_loss(logits, label)
                g_n = self.get_grad(loss, delta)
                grad_noise_sum += self._normalize_gradient(g_n)

                # ========== Branch3: hybrid (noise + dynamic block transform) ==========
                x_hyb = self.dynamic_block_transform(x_noisy)
                logits = self.get_logits(x_hyb)
                loss = self.get_loss(logits, label)
                g_hyb = self.get_grad(loss, delta)
                grad_hyb_sum += self._normalize_gradient(g_hyb)

            # Average over the copies & normalize each branch gradient
            grad_trans_avg = self._normalize_gradient(grad_trans_sum / self.copies)
            grad_noise_avg = self._normalize_gradient(grad_noise_sum / self.copies)
            grad_hyb_avg = self._normalize_gradient(grad_hyb_sum / self.copies)

            # Fuse the three branch gradients
            grad_final = (grad_trans_avg + grad_noise_avg + grad_hyb_avg) / 3.0
            grad_final = self._normalize_gradient(grad_final)

            # Momentum update
            momentum = self.get_momentum(grad_final, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()