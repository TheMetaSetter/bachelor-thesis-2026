from __future__ import annotations

"""Prototype memory helpers for the thesis multitask model."""

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.core.console import (
    console_print,
    debug_print,
    summarize_label_distribution,
)
from src.models.thesis_multitask_impl.thesis_multitask_state_memory_init_helpers import (
    move_initialization_batch_to_device as _move_initialization_batch_to_device,
    maybe_initialize_memories_from_loader as _maybe_initialize_memories_from_loader,
)


class ThesisMultitaskStateMemoryMixin:
    def _normalize_memory_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(vectors, dim=-1, eps=self.memory_norm_epsilon)

    def _normalize_hidden_for_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden, dim=-1, eps=self.memory_norm_epsilon)

    def _select_covering_vectors(
        self,
        candidate_vectors: torch.Tensor,
        num_vectors: int,
    ) -> torch.Tensor:
        if candidate_vectors.shape[0] == 0:
            raise ValueError("candidate_vectors must contain at least one token")

        normalized_vectors = self._normalize_memory_vectors(candidate_vectors)
        if normalized_vectors.shape[0] <= num_vectors:
            repeated_indices = (
                torch.arange(
                    num_vectors,
                    device=normalized_vectors.device,
                )
                % normalized_vectors.shape[0]
            )
            return normalized_vectors.index_select(0, repeated_indices)

        mean_vector = normalized_vectors.mean(dim=0, keepdim=True)
        squared_distances_to_mean = torch.sum(
            (normalized_vectors - mean_vector) ** 2,
            dim=1,
        )
        first_index = int(torch.argmin(squared_distances_to_mean).item())
        selected_indices = [first_index]
        minimum_squared_distances = torch.sum(
            (normalized_vectors - normalized_vectors[first_index]) ** 2,
            dim=1,
        )

        while len(selected_indices) < num_vectors:
            next_index = int(torch.argmax(minimum_squared_distances).item())
            selected_indices.append(next_index)
            next_squared_distances = torch.sum(
                (normalized_vectors - normalized_vectors[next_index]) ** 2,
                dim=1,
            )
            minimum_squared_distances = torch.minimum(
                minimum_squared_distances,
                next_squared_distances,
            )

        selected_index_tensor = torch.tensor(
            selected_indices,
            device=normalized_vectors.device,
        )
        return normalized_vectors.index_select(0, selected_index_tensor)

    def _run_kmeans(
        self,
        tokens: torch.Tensor,
        k: int,
        *,
        num_iterations: int,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [N, H]")
        if tokens.shape[0] == 0:
            raise ValueError("tokens must contain at least one row")
        if k <= 0:
            raise ValueError("k must be positive")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        normalized_tokens = self._normalize_memory_vectors(tokens)
        num_tokens = int(normalized_tokens.shape[0])
        debug_print(
            "MODEL",
            "Running memory bootstrap kmeans",
            token_count=num_tokens,
            centroid_count=k,
            num_iterations=num_iterations,
        )
        if num_tokens <= k:
            repeated_indices = (
                torch.arange(k, device=normalized_tokens.device) % num_tokens
            )
            return normalized_tokens.index_select(0, repeated_indices)

        token_mean = normalized_tokens.mean(dim=0, keepdim=True)
        squared_distances_to_mean = torch.sum(
            (normalized_tokens - token_mean) ** 2,
            dim=1,
        )
        first_index = int(torch.argmin(squared_distances_to_mean).item())
        selected_indices = [first_index]

        minimum_squared_distances = torch.sum(
            (normalized_tokens - normalized_tokens[first_index]) ** 2,
            dim=1,
        )
        while len(selected_indices) < k:
            next_index = int(torch.argmax(minimum_squared_distances).item())
            selected_indices.append(next_index)
            next_squared_distances = torch.sum(
                (normalized_tokens - normalized_tokens[next_index]) ** 2,
                dim=1,
            )
            minimum_squared_distances = torch.minimum(
                minimum_squared_distances,
                next_squared_distances,
            )

        centers = normalized_tokens.index_select(
            0,
            torch.tensor(selected_indices, device=normalized_tokens.device),
        )

        for _ in range(num_iterations):
            pairwise_distances = torch.cdist(normalized_tokens, centers, p=2)
            assignments = torch.argmin(pairwise_distances, dim=1)
            cluster_sizes = torch.bincount(assignments, minlength=k)
            empty_cluster_ids = torch.nonzero(
                cluster_sizes == 0, as_tuple=False
            ).flatten()
            if empty_cluster_ids.numel() > 0:
                debug_print(
                    "MODEL",
                    "Kmeans encountered empty clusters",
                    token_count=num_tokens,
                    centroid_count=k,
                    empty_cluster_count=int(empty_cluster_ids.numel()),
                    empty_cluster_ids=empty_cluster_ids.tolist(),
                    cluster_sizes=cluster_sizes.tolist(),
                )
            updated_centers: list[torch.Tensor] = []
            for center_index in range(k):
                cluster_mask = assignments == center_index
                if not torch.any(cluster_mask):
                    updated_centers.append(centers[center_index])
                    continue
                cluster_tokens = normalized_tokens[cluster_mask]
                updated_centers.append(cluster_tokens.mean(dim=0))
            centers = torch.stack(updated_centers, dim=0)
            centers = self._normalize_memory_vectors(centers)

        return centers

    def _collect_memory_initialization_token_pool_from_loader(
        self,
        train_loader: Any,
        device: str,
    ) -> dict[str, Any]:
        continuous_hidden_token_groups: list[torch.Tensor] = []
        discrete_hidden_tokens_by_class: dict[int, list[torch.Tensor]] = {}
        num_batches_used = 0
        previous_training_mode = self.training

        self.eval()
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(train_loader):
                if batch_index >= self.memory_initialization_batches:
                    break
                num_batches_used += 1
                batch_on_device = _move_initialization_batch_to_device(
                    raw_batch,
                    device,
                )
                clean_batch = self._prepare_clean_batch(
                    batch_on_device,
                    stage_name="memory_init",
                )
                clean_hidden = self.encoder(clean_batch)["hidden"].reshape(
                    -1,
                    self.hidden_dim,
                )
                if not (
                    self.memory_initialization_with_synthetic_windows
                    and self.use_synthetic_augmentation
                ):
                    debug_print(
                        "MODEL",
                        "Selected memory initialization batch",
                        batch_index=batch_index + 1,
                        batch_size=int(clean_windows.shape[0]),
                        class_distribution={"0": int(clean_windows.shape[0])},
                        synthetic_windows=0,
                        normal_windows=int(clean_windows.shape[0]),
                        train_balance_classes=bool(
                            getattr(self, "train_balance_classes", False)
                        ),
                        memory_initialization_with_synthetic_windows=bool(
                            self.memory_initialization_with_synthetic_windows
                        ),
                    )
                    continuous_hidden_token_groups.append(clean_hidden)
                    discrete_hidden_tokens_by_class.setdefault(0, []).append(
                        clean_hidden
                    )
                    continue

                synthetic_batch = self.synthetic_anomaly_injector.augment_batch(
                    self._clone_batch(batch_on_device)
                )
                synthetic_hidden = self.encoder(synthetic_batch)["hidden"]
                synthetic_labels = synthetic_batch["classification_labels"].long()
                synthetic_batch_size = int(synthetic_labels.shape[0])
                debug_print(
                    "MODEL",
                    "Selected memory initialization batch",
                    batch_index=batch_index + 1,
                    batch_size=synthetic_batch_size,
                    class_distribution=summarize_label_distribution(synthetic_labels),
                    synthetic_windows=int(
                        torch.count_nonzero(synthetic_labels != 0).detach().cpu()
                    ),
                    normal_windows=int(
                        torch.count_nonzero(synthetic_labels == 0).detach().cpu()
                    ),
                    train_balance_classes=bool(
                        getattr(self, "train_balance_classes", False)
                    ),
                    memory_initialization_with_synthetic_windows=bool(
                        self.memory_initialization_with_synthetic_windows
                    ),
                )
                normal_window_mask = synthetic_labels == 0
                normal_time_step_mask = synthetic_batch["synthetic_anomaly_mask"] == 0
                if int(normal_window_mask.sum().item()) > 0:
                    normal_hidden = synthetic_hidden[normal_window_mask]
                    normal_position_mask = normal_time_step_mask[normal_window_mask]
                    selected_normal_hidden = normal_hidden[normal_position_mask]
                    if selected_normal_hidden.numel() > 0:
                        continuous_hidden_token_groups.append(selected_normal_hidden)

                for class_index in synthetic_labels.unique(sorted=True).tolist():
                    class_mask = synthetic_labels == int(class_index)
                    class_hidden = synthetic_hidden[class_mask].reshape(
                        -1,
                        self.hidden_dim,
                    )
                    if class_hidden.numel() == 0:
                        continue
                    discrete_hidden_tokens_by_class.setdefault(
                        int(class_index), []
                    ).append(class_hidden)

        self.train(previous_training_mode)

        if continuous_hidden_token_groups:
            continuous_hidden_tokens = torch.cat(continuous_hidden_token_groups, dim=0)
        else:
            continuous_hidden_tokens = torch.empty(0, self.hidden_dim, device=device)

        finalized_discrete_hidden_tokens_by_class: dict[int, torch.Tensor] = {}
        for class_index, class_hidden_groups in discrete_hidden_tokens_by_class.items():
            finalized_discrete_hidden_tokens_by_class[class_index] = torch.cat(
                class_hidden_groups,
                dim=0,
            )

        return {
            "continuous_hidden_tokens": continuous_hidden_tokens,
            "discrete_hidden_tokens_by_class": finalized_discrete_hidden_tokens_by_class,
            "num_batches_used": num_batches_used,
            "num_continuous_normal_tokens": sum(
                int(hidden_group.shape[0])
                for hidden_group in continuous_hidden_token_groups
            ),
            "num_discrete_class_tokens_by_class": {
                class_index: int(class_hidden.shape[0])
                for class_index, class_hidden in finalized_discrete_hidden_tokens_by_class.items()
            },
        }

    def _initialize_memory_buffers_from_token_pool(
        self,
        *,
        continuous_hidden_tokens: torch.Tensor,
        discrete_hidden_tokens_by_class: dict[int, torch.Tensor],
    ) -> None:
        if continuous_hidden_tokens.shape[0] == 0:
            raise ValueError(
                "continuous_hidden_tokens must contain at least one normal token"
            )

        if self.continuous_prototype_bank is not None:
            continuous_seed_vectors = self._run_kmeans(
                continuous_hidden_tokens,
                self.continuous_num_prototypes,
                num_iterations=10,
            )
            self.continuous_prototype_bank.copy_(continuous_seed_vectors)

        if self.discrete_codebook is not None:
            available_class_indices = sorted(discrete_hidden_tokens_by_class)
            if not available_class_indices:
                raise ValueError(
                    "discrete_hidden_tokens_by_class must contain at least one class"
                )
            per_class_counts = [
                self.discrete_codebook_size // self.num_classes
                + (
                    1
                    if class_index < self.discrete_codebook_size % self.num_classes
                    else 0
                )
                for class_index in range(self.num_classes)
            ]
            fallback_hidden_tokens = torch.cat(
                [
                    discrete_hidden_tokens_by_class[class_index]
                    for class_index in available_class_indices
                ],
                dim=0,
            )
            class_stratified_vectors: list[torch.Tensor] = []
            for class_index, class_target_count in enumerate(per_class_counts):
                if class_target_count == 0:
                    continue
                class_hidden_tokens = discrete_hidden_tokens_by_class.get(class_index)
                if class_hidden_tokens is None or class_hidden_tokens.shape[0] == 0:
                    class_hidden_tokens = fallback_hidden_tokens
                class_stratified_vectors.append(
                    self._run_kmeans(
                        class_hidden_tokens,
                        class_target_count,
                        num_iterations=10,
                    )
                )
            discrete_seed_vectors = torch.cat(class_stratified_vectors, dim=0)
            if discrete_seed_vectors.shape[0] != self.discrete_codebook_size:
                raise ValueError(
                    "class-stratified discrete initialization must exactly fill "
                    f"discrete_codebook_size={self.discrete_codebook_size}, "
                    f"but produced {discrete_seed_vectors.shape[0]} vectors"
                )
            self.discrete_codebook.copy_(discrete_seed_vectors)
            if self.discrete_ema_counts is not None:
                self.discrete_ema_counts.fill_(1.0)
            if self.discrete_ema_sums is not None:
                self.discrete_ema_sums.copy_(discrete_seed_vectors)
            debug_print(
                "MODEL",
                "Initialized discrete memory buffers",
                available_class_indices=available_class_indices,
                per_class_counts=per_class_counts,
                codebook_size=int(self.discrete_codebook_size),
                seed_vector_sum=float(discrete_seed_vectors.sum().item()),
            )
            self._calibrate_anomaly_verification_metadata(
                discrete_hidden_tokens_by_class=discrete_hidden_tokens_by_class
            )

    # có cần thiết phải tính toán lại những thông tin metadata này hay không?
    # có thể tính toán ngay trong quá trình khởi tạo memory
    # sau khi stage A train xong có được không?
    def _calibrate_anomaly_verification_metadata(
        self, *, discrete_hidden_tokens_by_class: dict[int, torch.Tensor]
    ) -> None:
        # Nếu model không có discrete codebook thì không có gì để calibrate.
        if not isinstance(self.discrete_codebook, torch.Tensor):
            return

        # Lấy ra device hiện tại đang lưu codebook
        # để mọi tensor mới sinh ra nằm cùng nơi với codebook.
        codebook_device = self.discrete_codebook.device

        # Chia đều số codeword cho từng class.
        # Mỗi class nhận một đoạn liên tiếp trong codebook.
        counts = [
            self.discrete_codebook_size // self.num_classes
            # Nếu codebook không chia hết, các class đầu sẽ nhận thêm phần dư.
            + (1 if index < self.discrete_codebook_size % self.num_classes else 0)
            for index in range(self.num_classes)
        ]

        # mask: đánh dấu codeword nào được xem là anomalous.
        # codeword_class_ids: mỗi codeword thuộc class nào.
        # contributing_token_counts: có bao nhiêu token anomaly
        # đã "đóng góp" vào codeword đó,
        # hay nói cách khác: có bao nhiêu token là "cư dân"
        # trong cụm abstract anomalous pattern đó.

        # mỗi codeword là centroid của một cụm token đại diện cho
        # một abstract anomalous hoặc normal pattern.
        mask = torch.zeros(
            self.discrete_codebook_size, dtype=torch.bool, device=codebook_device
        )
        codeword_class_ids = torch.zeros(
            self.discrete_codebook_size, dtype=torch.long, device=codebook_device
        )
        contributing_token_counts = torch.zeros(
            self.discrete_codebook_size,
            dtype=torch.float32,
            device=codebook_device,
        )

        # Duyệt từng class và gán vùng codeword tương ứng.
        # Class 0 được xem như normal, các class > 0 được xem như anomaly.
        offset = 0
        for class_index, count in enumerate(counts):
            if class_index > 0:
                mask[offset : offset + count] = True
            codeword_class_ids[offset : offset + count] = class_index
            offset += count

        # Mặc định radius của mỗi codeword là 0.
        # Chỉ những codeword có anomaly token gắn vào mới được cập nhật radius thật.
        radii = torch.zeros(self.discrete_codebook_size, device=codebook_device)

        # Lấy toàn bộ hidden tokens thuộc các class anomaly (class > 0).
        anomaly_groups = [
            values.reshape(-1, self.hidden_dim)
            for class_index, values in discrete_hidden_tokens_by_class.items()
            if class_index > 0 and values.numel() > 0
        ]

        if anomaly_groups:
            # Gom tất cả anomaly tokens lại để đo khoảng cách với toàn bộ codebook.
            anomaly_tokens = torch.cat(anomaly_groups, dim=0).to(codebook_device)

            # Dùng cosine distance:
            # distance = 1 - cosine_similarity(token, codeword)
            distances = (
                1.0
                - F.normalize(anomaly_tokens, dim=-1)
                @ F.normalize(self.discrete_codebook, dim=-1).T
            )

            # Với mỗi anomaly token, tìm codeword gần nhất.
            nearest_ids = distances.argmin(dim=-1)
            nearest_distances = distances.gather(1, nearest_ids[:, None]).squeeze(1)

            # Đếm số token anomaly được gán cho từng codeword.
            contributing_token_counts += torch.bincount(
                nearest_ids,
                minlength=self.discrete_codebook_size,
            ).to(device=codebook_device, dtype=torch.float32)

            # Radius của mỗi codeword là quantile 0.99 của các token đã gán cho nó.
            # Ý nghĩa: lấy ngưỡng bao được phần lớn token anomaly gần codeword đó.
            for codeword_id in torch.unique(nearest_ids).tolist():
                assigned = nearest_distances[nearest_ids == codeword_id]
                radii[codeword_id] = torch.quantile(assigned, 0.99)
            negative_radii = radii[radii < 0]
            if negative_radii.numel() > 0:
                negative_entries = [
                    f"{int(codeword_id)}:{float(radius):.8e}"
                    for codeword_id, radius in zip(
                        torch.nonzero(radii < 0, as_tuple=False).flatten().tolist(),
                        negative_radii.tolist(),
                        strict=True,
                    )
                ]
                debug_print(
                    "MODEL",
                    "Found negative anomaly radii before clamping",
                    negative_count=int(negative_radii.numel()),
                    most_negative=float(negative_radii.min().item()),
                    negative_entries=negative_entries,
                )
            radii = radii.clamp_min(0.0)

        # Lưu toàn bộ metadata đã calibrate vào state của model.
        self.anomalous_codeword_mask = mask
        self.anomaly_radii = radii
        self.verification_codeword_class_ids = codeword_class_ids
        self.verification_contributing_token_counts = contributing_token_counts

        # Đánh dấu provenance của metadata này.
        # Nghĩa là: metadata được sinh từ luồng train anomaly tokens với q99.
        self.verification_metadata_source = "train_anomaly_tokens_q99"
        self.verification_metadata_schema_version = 1
        self.verification_metadata_split = "synthetic_train"

        # Lưu seed đã dùng để sinh metadata, ưu tiên synthetic_train_seed nếu có.
        self.verification_metadata_initialization_seed = int(
            self.synthetic_train_seed
            if getattr(self, "synthetic_train_seed", None) is not None
            else getattr(self, "synthetic_validation_seed", 0)
        )
        anomaly_token_count = int(
            sum(
                int(values.reshape(-1, self.hidden_dim).shape[0])
                for class_index, values in discrete_hidden_tokens_by_class.items()
                if class_index > 0 and values.numel() > 0
            )
        )
        mask_true_count = int(mask.sum().item())
        contributing_token_count_sum = int(contributing_token_counts.sum().item())
        if (
            mask_true_count == 0
            or contributing_token_count_sum == 0
            or contributing_token_count_sum != anomaly_token_count
        ):
            debug_print(
                "MODEL",
                "Verification metadata sanity check needs attention",
                anomaly_token_count=anomaly_token_count,
                mask_true_count=mask_true_count,
                contributing_token_count_sum=contributing_token_count_sum,
                expected_contributing_token_count=anomaly_token_count,
                codebook_size=int(self.discrete_codebook_size),
                radii_positive_count=int((radii > 0).sum().item()),
                class_token_counts={
                    str(class_index): int(values.reshape(-1, self.hidden_dim).shape[0])
                    for class_index, values in discrete_hidden_tokens_by_class.items()
                    if class_index > 0 and values.numel() > 0
                },
            )
        debug_print(
            "MODEL",
            "Calibrated verification metadata",
            mask_true_count=int(mask.sum().item()),
            radii_positive_count=int((radii > 0).sum().item()),
            radii_max=float(radii.max().item()) if radii.numel() > 0 else 0.0,
            codeword_class_ids_unique=sorted(
                {int(item) for item in codeword_class_ids.tolist()}
            ),
            contributing_token_count_sum=float(contributing_token_counts.sum().item()),
            verification_metadata_source=self.verification_metadata_source,
        )

    def _update_continuous_memory_bank(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.continuous_prototype_bank is None:
            raise ValueError("continuous_prototype_bank is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                return self._normalize_memory_vectors(self.continuous_prototype_bank)
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        normalized_memory = self._normalize_memory_vectors(
            self.continuous_prototype_bank
        )

        prototype_to_token_logits = torch.einsum(
            "kh,blh->kbl",
            normalized_memory,
            normalized_hidden,
        ) / math.sqrt(self.hidden_dim)

        prototype_to_token_weights = torch.softmax(
            prototype_to_token_logits.reshape(self.continuous_num_prototypes, -1),
            dim=-1,
        ).reshape_as(prototype_to_token_logits)

        weighted_hidden_summary = torch.einsum(
            "kbl,blh->kh",
            prototype_to_token_weights,
            normalized_hidden,
        )

        weighted_hidden_summary = self._normalize_memory_vectors(
            weighted_hidden_summary
        )

        gate_input = torch.cat(
            [normalized_memory, weighted_hidden_summary],
            dim=-1,
        )

        update_gate = self.continuous_update_gate(gate_input)

        updated_memory = (
            1.0 - update_gate
        ) * normalized_memory + update_gate * weighted_hidden_summary
        updated_memory = self._normalize_memory_vectors(updated_memory)

        with torch.no_grad():
            self.continuous_prototype_bank.copy_(updated_memory.detach())

        return updated_memory

    def _update_discrete_codebook_memory(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.discrete_assignment is None
            or self.discrete_codebook is None
            or self.discrete_ema_counts is None
            or self.discrete_ema_sums is None
        ):
            raise ValueError("discrete memory state is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                assignment_logits = hidden.new_zeros(
                    hidden.shape[0],
                    hidden.shape[1],
                    self.discrete_codebook_size,
                )
                assignment_probabilities = torch.softmax(assignment_logits, dim=-1)
                return (
                    assignment_logits,
                    assignment_probabilities,
                    self._normalize_memory_vectors(self.discrete_codebook),
                )
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        assignment_logits = self.discrete_assignment(normalized_hidden)
        assignment_probabilities = F.gumbel_softmax(
            assignment_logits,
            tau=self.gumbel_temperature,
            hard=False,
            dim=-1,
        )
        flattened_probabilities = assignment_probabilities.reshape(
            -1,
            self.discrete_codebook_size,
        )
        flattened_hidden = normalized_hidden.reshape(-1, self.hidden_dim)
        batch_counts = flattened_probabilities.sum(dim=0)
        batch_sums = flattened_probabilities.T @ flattened_hidden

        with torch.no_grad():
            self.discrete_ema_counts.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_counts.detach()
            )
            self.discrete_ema_sums.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_sums.detach()
            )
            normalized_codebook = (
                self.discrete_ema_sums
                / self.discrete_ema_counts.clamp_min(
                    self.memory_norm_epsilon
                ).unsqueeze(-1)
            )
            normalized_codebook = self._normalize_memory_vectors(normalized_codebook)
            self.discrete_codebook.copy_(normalized_codebook)

        return (
            assignment_logits,
            assignment_probabilities,
            self._normalize_memory_vectors(self.discrete_codebook),
        )

    def maybe_initialize_memories_from_loader(
        self,
        train_loader: Any,
        *,
        device: str,
    ) -> bool:
        return _maybe_initialize_memories_from_loader(
            self,
            train_loader,
            device=device,
        )
