import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import random
from itertools import combinations


class AxisIdAndValueEmbedder(nn.Module):
    def __init__(
        self, num_axes: int, cond_dim: int, value_mlp_hidden: int, axisembed_method: str
    ):

        super().__init__()
        self.axisembed_method = axisembed_method
        self.num_axes = num_axes
        self.cond_dim = cond_dim

        if axisembed_method == "concat":

            self.single_axis_dim = cond_dim // num_axes
        else:

            self.single_axis_dim = cond_dim

        if axisembed_method == "combinatorial":

            self.combo_to_idx = {}
            idx = 0
            for depth in range(1, num_axes + 1):
                for axis_combo in combinations(range(num_axes), depth):
                    for signs in range(2**depth):

                        sign_list = [
                            (1 if (signs >> i) & 1 else -1) for i in range(depth)
                        ]

                        key = tuple((axis_combo[i], sign_list[i]) for i in range(depth))
                        self.combo_to_idx[key] = idx
                        idx += 1

            self.num_combos = 3**num_axes - 1
            assert (
                idx == self.num_combos
            ), f"Expected {self.num_combos} combinations, got {idx}"
            self.axis_id_to_embedder = nn.Embedding(
                self.num_combos, self.single_axis_dim
            )
            self.axis_value_to_d_mlp = None
        else:
            self.axis_id_to_embedder = nn.Embedding(num_axes, self.single_axis_dim)
            self.axis_value_to_d_mlp = nn.Sequential(
                nn.Linear(1, value_mlp_hidden),
                nn.GELU(),
                nn.Linear(value_mlp_hidden, self.single_axis_dim),
            )

        self.debug_counter = 0

    def _get_combinatorial_terms_for_item(
        self,
        axis_id_row: torch.Tensor,
        axis_value_row: torch.Tensor,
        axis_mask_row: torch.Tensor,
        combinatorial_max_order: Optional[int],
    ) -> List[Tuple[int, float]]:

        max_num_axes_in_batch = int(axis_id_row.shape[0])
        active_axes_and_values = [
            (
                int(axis_id_row[axis_id_iterator].item()),
                float(axis_value_row[axis_id_iterator].item()),
            )
            for axis_id_iterator in range(max_num_axes_in_batch)
            if bool(axis_mask_row[axis_id_iterator].item())
        ]
        if not active_axes_and_values:
            return []

        active_axes_and_values.sort(key=lambda x: x[0])
        max_depth = len(active_axes_and_values)
        if combinatorial_max_order is not None:
            max_depth = min(max_depth, combinatorial_max_order)

        out_terms: List[Tuple[int, float]] = []
        for depth in range(1, max_depth + 1):
            for subset_indices in combinations(
                range(len(active_axes_and_values)), depth
            ):
                subset_axes_and_values = [
                    active_axes_and_values[i] for i in subset_indices
                ]
                list_axes_and_direction = []
                weight = 1.0
                for axis_id_val, axis_val in subset_axes_and_values:
                    sign = 1 if axis_val >= 0 else -1
                    list_axes_and_direction.append((axis_id_val, sign))
                    weight *= abs(axis_val)
                embedding_key = tuple(list_axes_and_direction)
                embedding_combo_idx = self.combo_to_idx[embedding_key]
                out_terms.append((embedding_combo_idx, weight))
        return out_terms

    def forward_combinatorial_tokens(
        self,
        axis_id: torch.Tensor,
        axis_value: torch.Tensor,
        axis_mask: torch.Tensor,
        combinatorial_max_order: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.axisembed_method != "combinatorial":
            raise ValueError(
                "forward_combinatorial_tokens requires axisembed_method='combinatorial'"
            )
        assert axis_id.dim() == 2

        num_items_in_batch = int(axis_id.shape[0])
        per_item_terms: List[List[Tuple[int, float]]] = []
        max_terms = 0
        for item_id in range(num_items_in_batch):
            terms = self._get_combinatorial_terms_for_item(
                axis_id[item_id],
                axis_value[item_id],
                axis_mask[item_id],
                combinatorial_max_order=combinatorial_max_order,
            )
            per_item_terms.append(terms)
            max_terms = max(max_terms, len(terms))

        if max_terms == 0:

            tokens = torch.zeros(
                num_items_in_batch,
                1,
                self.single_axis_dim,
                dtype=self.axis_id_to_embedder.weight.dtype,
                device=axis_id.device,
            )
            token_mask = torch.zeros(
                num_items_in_batch, 1, dtype=torch.bool, device=axis_id.device
            )
            return tokens, token_mask

        tokens = torch.zeros(
            num_items_in_batch,
            max_terms,
            self.single_axis_dim,
            dtype=self.axis_id_to_embedder.weight.dtype,
            device=axis_id.device,
        )
        token_mask = torch.zeros(
            num_items_in_batch, max_terms, dtype=torch.bool, device=axis_id.device
        )

        for item_id, terms in enumerate(per_item_terms):
            if not terms:
                continue
            combo_indices = torch.tensor(
                [combo_idx for combo_idx, _ in terms],
                dtype=torch.long,
                device=axis_id.device,
            )
            weights = torch.tensor(
                [weight for _, weight in terms],
                dtype=tokens.dtype,
                device=axis_id.device,
            ).unsqueeze(-1)
            tokens[item_id, : len(terms), :] = (
                self.axis_id_to_embedder(combo_indices) * weights
            )
            token_mask[item_id, : len(terms)] = True

        return tokens, token_mask

    def forward(
        self,
        axis_id: torch.Tensor,
        axis_value: torch.Tensor,
        axis_mask: torch.Tensor,
        combinatorial_max_order: Optional[int] = None,
    ) -> torch.Tensor:

        assert axis_id.dim() == 2

        axis_id_embedded = self.axis_id_to_embedder(axis_id)

        if self.axisembed_method == "combinatorial":

            num_items_in_batch, max_num_axes_in_batch = axis_id.shape
            output_conditioning_for_batch = torch.zeros(
                num_items_in_batch,
                self.single_axis_dim,
                dtype=axis_id_embedded.dtype,
                device=axis_id.device,
            )

            for item_id in range(num_items_in_batch):
                terms = self._get_combinatorial_terms_for_item(
                    axis_id[item_id],
                    axis_value[item_id],
                    axis_mask[item_id],
                    combinatorial_max_order=combinatorial_max_order,
                )
                for embedding_combo_idx, weight in terms:
                    output_conditioning_for_batch[item_id] += (
                        weight * self.axis_id_to_embedder.weight[embedding_combo_idx]
                    )

            output_conditioning_for_batch = output_conditioning_for_batch / float(
                self.num_combos
            )

            return output_conditioning_for_batch

        elif self.axisembed_method == "addall":
            axis_value_embedded = self.axis_value_to_d_mlp(axis_value.view(-1, 1)).view(
                axis_id_embedded.shape
            )
            output_conditioning_for_batch = axis_id_embedded + axis_value_embedded

        elif self.axisembed_method == "mult":

            output_conditioning_for_batch = axis_id_embedded * axis_value.unsqueeze(
                -1
            ).to(dtype=axis_id_embedded.dtype)
        elif self.axisembed_method == "concat":

            num_items_in_batch, max_num_axes_in_batch = axis_id.shape

            cond_concat = torch.zeros(
                num_items_in_batch,
                self.num_axes * self.single_axis_dim,
                dtype=axis_id_embedded.dtype,
                device=axis_id_embedded.device,
            )

            for k in range(max_num_axes_in_batch):

                axis_k_embedded = axis_id_embedded[:, k, :]
                axis_k_value = axis_value[:, k].unsqueeze(-1)
                axis_k_cond = axis_k_embedded * axis_k_value

                axis_k_mask = axis_mask[:, k].unsqueeze(-1).to(dtype=axis_k_cond.dtype)
                axis_k_cond = axis_k_cond * axis_k_mask

                for item_id in range(num_items_in_batch):

                    target_axis = int(axis_id[item_id, k].item())

                    start_idx = target_axis * self.single_axis_dim
                    end_idx = (target_axis + 1) * self.single_axis_dim
                    cond_concat[item_id, start_idx:end_idx] = axis_k_cond[item_id, :]

            return cond_concat
        else:
            raise NotImplementedError(
                f"Unknown axisembed_method={self.axisembed_method}"
            )

        axis_mask_value = axis_mask.to(
            dtype=output_conditioning_for_batch.dtype
        ).unsqueeze(-1)

        output_conditioning_for_batch = (
            output_conditioning_for_batch * axis_mask_value
        ).sum(dim=1)

        if False:
            denom = axis_mask_value.sum(dim=1).clamp(min=1.0)
            output_conditioning_for_batch = output_conditioning_for_batch / denom

        else:

            output_conditioning_for_batch = output_conditioning_for_batch / float(
                self.num_axes
            )

        return output_conditioning_for_batch


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0)


class FiLM(nn.Module):

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, d_model)
        self.to_beta = nn.Linear(cond_dim, d_model)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

        gamma = self.to_gamma(c).unsqueeze(1)
        beta = self.to_beta(c).unsqueeze(1)
        return gamma * h + beta


class AdaLayerNorm(nn.Module):

    def __init__(self, d_model: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.to_gamma = nn.Linear(cond_dim, d_model)
        self.to_beta = nn.Linear(cond_dim, d_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

        x_mu = x.mean(dim=-1, keepdim=True)
        x_var = (x - x_mu).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - x_mu) / torch.sqrt(x_var + self.eps)

        gamma = self.to_gamma(c).unsqueeze(1)
        beta = self.to_beta(c).unsqueeze(1)

        return x_norm * (1.0 + gamma) + beta


class AdaLayerNorm2(nn.Module):

    def __init__(self, d_model: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.to_gamma = nn.Linear(cond_dim, d_model)
        self.to_beta = nn.Linear(cond_dim, d_model)
        self.to_delta = nn.Linear(d_model + cond_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        x_mu = x.mean(dim=-1, keepdim=True)
        x_var = (x - x_mu).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - x_mu) / torch.sqrt(x_var + self.eps)

        gamma = self.to_gamma(c).unsqueeze(1)
        beta = self.to_beta(c).unsqueeze(1)
        x_mod = x_norm * (1.0 + gamma) + beta

        c_expanded = c.unsqueeze(1).expand(-1, x_mod.shape[1], -1)
        delta = self.to_delta(torch.cat([x_mod, c_expanded], dim=-1))
        return x_mod + delta


class CondTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        cond_dim: int,
        cond_method: str,
    ):
        super().__init__()
        self.cond_method = cond_method

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.cross_attn = None
        self.cross_dropout = None
        self.cross_norm = None

        if cond_method == "adaln":
            self.norm1 = AdaLayerNorm(d_model, cond_dim)
            self.norm2 = AdaLayerNorm(d_model, cond_dim)
            self.film1 = None
            self.film2 = None
        elif cond_method == "adaln2":
            self.norm1 = AdaLayerNorm2(d_model, cond_dim)
            self.norm2 = AdaLayerNorm2(d_model, cond_dim)
            self.film1 = None
            self.film2 = None
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            if cond_method == "film":
                self.film1 = FiLM(d_model, cond_dim)
                self.film2 = FiLM(d_model, cond_dim)
            else:
                self.film1 = None
                self.film2 = None
            if cond_method == "cross":
                self.cross_attn = nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                )
                self.cross_dropout = nn.Dropout(dropout)
                self.cross_norm = nn.LayerNorm(d_model)

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        desired_axis_embedding: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        desired_axis_tokens: Optional[torch.Tensor] = None,
        desired_axis_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        x = x + self.dropout1(attn_out)

        if self.cond_method in {"adaln", "adaln2"}:

            x = self.norm1(x, desired_axis_embedding)
        else:
            x = self.norm1(x)
            if self.film1 is not None:
                x = self.film1(x, desired_axis_embedding)

        if self.cond_method == "cross":
            assert self.cross_attn is not None
            assert self.cross_dropout is not None
            assert self.cross_norm is not None
            assert desired_axis_tokens is not None
            assert desired_axis_tokens_mask is not None
            cond_key_padding_mask = ~desired_axis_tokens_mask
            cross_out, _ = self.cross_attn(
                x,
                desired_axis_tokens,
                desired_axis_tokens,
                key_padding_mask=cond_key_padding_mask,
                need_weights=False,
            )
            x = self.cross_norm(x + self.cross_dropout(cross_out))

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)

        if self.cond_method in {"adaln", "adaln2"}:
            x = self.norm2(x, desired_axis_embedding)
        else:
            x = self.norm2(x)
            if self.film2 is not None:
                x = self.film2(x, desired_axis_embedding)

        return x


class CondTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        cond_dim: int,
        cond_method: str,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CondTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    cond_dim=cond_dim,
                    cond_method=cond_method,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        points_embedding: torch.Tensor,
        desired_axis_embedding: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        desired_axis_tokens: Optional[torch.Tensor] = None,
        desired_axis_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            points_embedding = layer(
                points_embedding,
                desired_axis_embedding,
                src_key_padding_mask=src_key_padding_mask,
                desired_axis_tokens=desired_axis_tokens,
                desired_axis_tokens_mask=desired_axis_tokens_mask,
            )
        return points_embedding


class GlyphDeltaRegressor(nn.Module):
    def __init__(
        self,
        num_axes: int,
        max_points: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        cond_method: str,
        pos_enc: str,
        point_mlp_hidden: int,
        cond_mlp_hidden: int,
        point_contour_info: int,
        num_contour_id_plus_three: int,
        point_feat_dim: int,
        axisembed_method: str,
        polar_regression: bool = False,
        coordnorm: bool = False,
    ):
        super().__init__()

        self.num_axes = num_axes
        self.cond_method = cond_method
        self.axisembed_method = axisembed_method
        self.polar_regression = bool(polar_regression)
        self.coordnorm = bool(coordnorm)
        self.point_contour_info = int(point_contour_info)
        self.axis_cond_dim = (
            d_model * num_axes if axisembed_method == "concat" else d_model
        )

        self.point_mlp = nn.Sequential(
            nn.Linear(point_feat_dim, point_mlp_hidden),
            nn.GELU(),
            nn.Linear(point_mlp_hidden, d_model),
        )

        self.positional_encoding = (
            SinusoidalPosEnc(
                d_model, max_len=max_points + (1 if cond_method == "token" else 0)
            )
            if pos_enc == "sinusoidal"
            else None
        )

        axis_cond_dim = self.axis_cond_dim

        self.axis_id_and_value_embedder = AxisIdAndValueEmbedder(
            num_axes=num_axes,
            cond_dim=axis_cond_dim,
            value_mlp_hidden=cond_mlp_hidden,
            axisembed_method=axisembed_method,
        )

        self.contour_embedder = None
        if self.point_contour_info == 1:
            self.contour_embedder = nn.Embedding(num_contour_id_plus_three, d_model)
        else:

            pass

        if cond_method == "token":
            encoder_cond_method = "none"
        elif cond_method == "film":
            encoder_cond_method = "film"
        elif cond_method == "adaln":
            encoder_cond_method = "adaln"
        elif cond_method == "adaln2":
            encoder_cond_method = "adaln2"
        elif cond_method == "cross":
            encoder_cond_method = "cross"
        else:
            raise ValueError(f"Unknown cond_method={cond_method}")

        self.cond_transformer_encoder = CondTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            cond_dim=axis_cond_dim,
            cond_method=encoder_cond_method,
        )

        delta_out_dim = 3 if self.polar_regression else 2
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, delta_out_dim),
        )

        if cond_method == "cross":
            if axisembed_method != "combinatorial":
                raise ValueError(
                    "cond_method='cross' requires axisembed_method='combinatorial'"
                )

    def forward(
        self,
        points: torch.Tensor,
        contour_id: torch.Tensor,
        desired_axis_id: torch.Tensor,
        desired_axis_value: torch.Tensor,
        pad_mask: torch.Tensor,
        desired_axis_mask: torch.Tensor,
        combinatorial_max_order: Optional[int] = None,
    ) -> torch.Tensor:

        B, N, _ = points.shape
        points_for_model = points
        real_base_mask = (~pad_mask) & (points[:, :, 2] >= 0.0)
        per_example_scale: Optional[torch.Tensor] = None

        if self.coordnorm:

            xy = points[:, :, :2]
            x = xy[:, :, 0]
            y = xy[:, :, 1]
            inf = torch.finfo(xy.dtype).max
            ninf = -inf
            has_real = real_base_mask.any(dim=1)

            x_min = torch.where(real_base_mask, x, inf).amin(dim=1)
            x_max = torch.where(real_base_mask, x, ninf).amax(dim=1)
            y_min = torch.where(real_base_mask, y, inf).amin(dim=1)
            y_max = torch.where(real_base_mask, y, ninf).amax(dim=1)

            zero = torch.zeros_like(x_min)
            one = torch.ones_like(x_min)
            cx = torch.where(has_real, 0.5 * (x_min + x_max), zero)
            cy = torch.where(has_real, 0.5 * (y_min + y_max), zero)
            w = torch.where(has_real, x_max - x_min, one)
            h = torch.where(has_real, y_max - y_min, one)
            s = torch.maximum(w, h).clamp(min=1e-6)

            c = torch.stack([cx, cy], dim=-1).unsqueeze(1)
            s_expanded = s.view(B, 1, 1)
            xy_norm = 2.0 * (xy - c) / s_expanded
            xy_out = torch.where(real_base_mask.unsqueeze(-1), xy_norm, xy)

            points_for_model = points.clone()
            points_for_model[:, :, :2] = xy_out
            per_example_scale = s_expanded

        desired_axis_embedding: Optional[torch.Tensor] = None
        desired_axis_tokens: Optional[torch.Tensor] = None
        desired_axis_tokens_mask: Optional[torch.Tensor] = None
        if self.cond_method == "cross":
            desired_axis_tokens, desired_axis_tokens_mask = (
                self.axis_id_and_value_embedder.forward_combinatorial_tokens(
                    desired_axis_id,
                    desired_axis_value,
                    desired_axis_mask,
                    combinatorial_max_order=combinatorial_max_order,
                )
            )
        else:
            desired_axis_embedding = self.axis_id_and_value_embedder(
                desired_axis_id,
                desired_axis_value,
                desired_axis_mask,
                combinatorial_max_order=combinatorial_max_order,
            )

        points_embedding = self.point_mlp(points_for_model)

        if self.contour_embedder is not None:
            points_embedding = points_embedding + self.contour_embedder(contour_id)

        if self.cond_method == "token":
            assert desired_axis_embedding is not None
            c_tok = desired_axis_embedding.unsqueeze(1)
            points_embedding = torch.cat([c_tok, points_embedding], dim=1)
            pad_mask = torch.cat(
                [
                    torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device),
                    pad_mask,
                ],
                dim=1,
            )

        if self.positional_encoding is not None:
            points_embedding = self.positional_encoding(points_embedding)

        points_embedding = self.cond_transformer_encoder(
            points_embedding,
            (
                desired_axis_embedding
                if desired_axis_embedding is not None
                else points_embedding.new_zeros((B, self.axis_cond_dim))
            ),
            src_key_padding_mask=pad_mask,
            desired_axis_tokens=desired_axis_tokens,
            desired_axis_tokens_mask=desired_axis_tokens_mask,
        )

        if self.cond_method == "token":
            points_embedding = points_embedding[:, 1:, :]

        pred = self.delta_head(points_embedding)

        if not self.polar_regression:
            if self.coordnorm and per_example_scale is not None:

                pred = torch.where(
                    real_base_mask.unsqueeze(-1), pred * per_example_scale, pred
                )
            return pred
        else:

            r = pred[..., 0:1]
            c = pred[..., 1:2]
            s = pred[..., 2:3]
            norm = (c.square() + s.square()).clamp(min=1e-12).sqrt()
            c_unit = c / norm
            s_unit = s / norm
            dx = r * c_unit
            dy = r * s_unit
            pred_cartesian = torch.cat([dx, dy], dim=-1)
            if self.coordnorm and per_example_scale is not None:
                pred_cartesian = torch.where(
                    real_base_mask.unsqueeze(-1),
                    pred_cartesian * per_example_scale,
                    pred_cartesian,
                )
            return pred_cartesian


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta, reduction="none")
