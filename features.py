import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops


#===== Gather functions

def gather_edges(edge_tensor, neighbor_idx):
    # features [B, N, N , C] n_idx [B, N, K] -> neighbor features [B, N, K, C]

    idx = neighbor_idx[..., None]                        # (B, N, K, 1)
    idx = jnp.broadcast_to(idx, (*neighbor_idx.shape, edge_tensor.shape[-1]))  # (B, N, K, C)
    edge_features = jnp.take_along_axis(edge_tensor, idx, axis=2)
    return edge_features

def gather_nodes(node_tensor, neighbor_idx):
    # nodes: [B, N, C], neighbor_idx: [B, N, K] -> [B, N, K, C]
    b, n, k = neighbor_idx.shape
    c = node_tensor.shape[-1]
    idx = einops.rearrange(neighbor_idx, 'b n k -> b (n k)')   # [B, NK]
    idx = idx[..., None]                                        # [B, NK, 1]
    idx = jnp.broadcast_to(idx, (b, n * k, c))                 # [B, NK, C]
    neighbor_features = jnp.take_along_axis(node_tensor, idx, axis=1) # [B, NK, C]
    neighbor_features = einops.rearrange(
        neighbor_features, 'b (n k) c -> b n k c', n=n, k=k
    )
    return neighbor_features

def gather_nodes_t(node_tensor, neighbor_idx):
    # features [B, N, C] a N_idx [B, K] -> neighbor_features [B, K, C]
    b, n, c = node_tensor.shape
    _, k = neighbor_idx.shape
    idx_flat = neighbor_idx[..., None]
    idx_flat = jnp.broadcast_to(idx_flat, (b, k, c))
    neighbor_features = jnp.take_along_axis(node_tensor, idx_flat, axis=1)
    return neighbor_features


#=== Position Layers
class PositionalEncodings(nnx.Module):
    def __init__(self, num_embeddings, rngs: nnx.Rngs, max_relative_feature=32)-> None:
        kernel_init = jax.nn.initializers.glorot_uniform()
        bias_init = jax.nn.initializers.zeros
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nnx.Linear(
            2*max_relative_feature+1+1, num_embeddings, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )

    def __call__(self, residue_offset, same_chain_mask):
        d = jnp.clip(residue_offset + self.max_relative_feature, 0, 2*self.max_relative_feature) * same_chain_mask
        d_onehot = jax.nn.one_hot(d, num_classes=2*self.max_relative_feature+1+1)
        positional_features = self.linear(d_onehot)
        return positional_features

class CA_ProteinFeatures(nnx.Module):
    def __init__(self, edge_features, node_features, rngs:nnx.Rngs, num_positional_embeddings=16 ,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16)-> None:
        kernel_init = jax.nn.initializers.glorot_uniform()

        '''Extract Protein features'''
        self.edge_features = edge_features
        self.node_features= node_features
        self.num_chain_embeddings= num_chain_embeddings
        self.num_rbf= num_rbf
        self.top_k= top_k
        self.augment_eps= augment_eps
        self.num_chain_embeddings= num_chain_embeddings

        #positional encoding
        self.embeddings = PositionalEncodings(num_chain_embeddings, rngs=rngs)
        #normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf*9 + 7
        self.node_embedding = nnx.Linear(
            node_in, node_features, use_bias=False, rngs=rngs, kernel_init=kernel_init
        )
        self.edge_embedding = nnx.Linear(
            edge_in, edge_features, use_bias=False, rngs=rngs, kernel_init=kernel_init
        )
        self.norm_nodes = nnx.LayerNorm(node_features, rngs=rngs)
        self.norm_edges = nnx.LayerNorm(edge_features, rngs=rngs)

    def _quaternions(self, R):
        '''Convert a batch of 3D rotations [R] to quaterions [Q],
            R [..., 3, 3]
            Q = [..., 4]
        '''
        diag = jnp.diagonal(R, axis1=-2, axis2=-1)
        Rxx, Ryy, Rzz = diag[...,0] , diag[...,1], diag[...,2]

        magnitudes = .5 * jnp.sqrt(jnp.abs(1 + jnp.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], axis=-1)))

        _R = lambda i, j: R[..., i, j]
        signs = jnp.sign(jnp.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1),
        ], axis=-1))
        xyz = signs * magnitudes
        diag = jnp.diagonal(R, axis1=-2, axis2=-1)  # [...,3]
        # relu -> jnp.maximum(., 0)
        w = jnp.sqrt(jnp.maximum(1.0 + diag.sum(axis=-1, keepdims=True), 0.0)) / 2.0
        Q = jnp.concatenate([xyz, w], axis=-1)
        # F.normalize(Q, dim=-1)
        eps = jnp.finfo(Q.dtype).eps
        Q = Q / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + eps)
        return Q
    def _normalize(self, x, axis=-1, eps=1e-8):
        return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)

    def _orientations_coarse(self, ca_coords, neighbor_idx, eps=1e-6):
        # ca_coords: [B, N, 3], neighbor_idx: [B, N, K]
        dX = ca_coords[:, 1:, :] - ca_coords[:, :-1, :]                         # [B, N-1, 3]
        dX_norm = jnp.linalg.norm(dX, axis=-1)                  # [B, N-1]
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)             # [B, N-1]
        dX = dX * dX_mask[..., None]
        U = self._normalize(dX, axis=-1)                             # [B, N-1, 3]
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = self._normalize(jnp.cross(u_2, u_1, axis=-1), axis=-1)
        n_1 = self._normalize(jnp.cross(u_1, u_0, axis=-1), axis=-1)
        # Bond angle
        cosA = -jnp.sum(u_1 * u_0, axis=-1)
        cosA = jnp.clip(cosA, -1.0 + eps, 1.0 - eps)
        A = jnp.arccos(cosA)
        # Dihedral
        cosD = jnp.sum(n_2 * n_1, axis=-1)
        cosD = jnp.clip(cosD, -1.0 + eps, 1.0 - eps)
        D = jnp.sign(jnp.sum(u_2 * n_1, axis=-1)) * jnp.arccos(cosD)
        # AD features: [B, N-3, 3] -> pad to [B, N, 3]
        AD_features = jnp.stack(
            [jnp.cos(A), jnp.sin(A) * jnp.cos(D), jnp.sin(A) * jnp.sin(D)],
            axis=2
        )
        AD_features = jnp.pad(AD_features, ((0, 0), (1, 2), (0, 0)), mode="constant")
        # Local frames
        o_1 = self._normalize(u_2 - u_1, axis=-1)
        O = jnp.stack([o_1, n_2, jnp.cross(o_1, n_2, axis=-1)], axis=2)  # [B, N-3, 3, 3]
        O = O.reshape(*O.shape[:2], 9)                                    # [B, N-3, 9]
        O = jnp.pad(O, ((0, 0), (1, 2), (0, 0)), mode="constant")         # [B, N, 9]
        O_neighbors = gather_nodes(O, neighbor_idx)          # [B, N, K, 9]
        X_neighbors = gather_nodes(ca_coords, neighbor_idx)          # [B, N, K, 3]
        O = O.reshape(*O.shape[:2], 3, 3)             # [B, N, 3, 3]
        O_neighbors = O_neighbors.reshape(*O_neighbors.shape[:3], 3, 3)  # [B, N, K, 3, 3]
        # Relative orientation
        dX = X_neighbors - ca_coords[:, :, None, :]           # [B, N, K, 3]
        dU = jnp.matmul(O[:, :, None, :, :], dX[..., None]).squeeze(-1)   # [B, N, K, 3]
        dU = self._normalize(dU, axis=-1)
        R = jnp.matmul(jnp.swapaxes(O[:, :, None, :, :], -1, -2), O_neighbors)  # [B,N,K,3,3]
        Q = self._quaternions(R)                      # [B, N, K, 4]
        O_features = jnp.concatenate([dU, Q], axis=-1)  # [B, N, K, 7]
        return AD_features, O_features
    def _dist(self, ca_coords, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = ca_coords[:, None, :, :] - ca_coords[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)

        # Identify k nearest neighbors (including self)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1. - mask_2D) * D_max
        k = min(self.top_k, ca_coords.shape[1])
        neighbor_idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        neighbor_distances = jnp.take_along_axis(D_adjust, neighbor_idx, axis=-1)
        neighbor_mask = gather_edges(mask_2D[..., None], neighbor_idx)
        return neighbor_distances, neighbor_idx, neighbor_mask

    def _rbf(self, distances):
        # Distance radial basis function
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape((1, 1, 1, -1))
        D_sigma = (D_max - D_min) / D_count
        D_expand = distances[..., None]
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, atom_coords_a, atom_coords_b, neighbor_idx):
        pairwise_distances = jnp.sqrt(jnp.sum((atom_coords_a[:, :, None, :] - atom_coords_b[:, None, :, :])**2, axis=-1) + 1e-6) #[B, L, L]
        neighbor_distances = gather_edges(pairwise_distances[..., None], neighbor_idx)[..., 0] #[B,L,K]
        rbf_features = self._rbf(neighbor_distances)
        return rbf_features

    def __call__(self, ca_coords, mask, residue_idx, chain_labels, noise_key=None):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0 and noise_key is not None:
            ca_coords = ca_coords + self.augment_eps * jax.random.normal(noise_key, shape=ca_coords.shape, dtype=ca_coords.dtype)

        neighbor_distances, neighbor_idx, neighbor_mask = self._dist(ca_coords, mask)

        ca_prev = jnp.zeros_like(ca_coords)
        ca_next = jnp.zeros_like(ca_coords)
        ca_prev = ca_prev.at[:, 1:, :].set(ca_coords[:, :-1, :])
        ca_curr = ca_coords
        ca_next = ca_next.at[:, :-1, :].set(ca_coords[:, 1:, :])

        node_orientation_features, edge_orientation_features = self._orientations_coarse(ca_coords, neighbor_idx)

        rbf_feature_blocks = []
        rbf_feature_blocks.append(self._rbf(neighbor_distances)) #Ca_1-Ca_1
        rbf_feature_blocks.append(self._get_rbf(ca_prev, ca_prev, neighbor_idx))
        rbf_feature_blocks.append(self._get_rbf(ca_next, ca_next, neighbor_idx))

        rbf_feature_blocks.append(self._get_rbf(ca_prev, ca_curr, neighbor_idx))
        rbf_feature_blocks.append(self._get_rbf(ca_prev, ca_next, neighbor_idx))

        rbf_feature_blocks.append(self._get_rbf(ca_curr, ca_prev, neighbor_idx))
        rbf_feature_blocks.append(self._get_rbf(ca_curr, ca_next, neighbor_idx))

        rbf_feature_blocks.append(self._get_rbf(ca_next, ca_prev, neighbor_idx))
        rbf_feature_blocks.append(self._get_rbf(ca_next, ca_curr, neighbor_idx))


        rbf_all = jnp.concatenate(rbf_feature_blocks, axis=-1)


        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], neighbor_idx)[:, :, :, 0] #[B, L, K]

        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        edge_chain_mask = gather_edges(d_chains[:, :, :, None], neighbor_idx)[:, :, :, 0]
        edge_positional_features = self.embeddings(offset.astype(jnp.int32), edge_chain_mask)
        edge_features = jnp.concatenate((edge_positional_features, rbf_all, edge_orientation_features), axis=-1)


        edge_features = self.edge_embedding(edge_features)
        edge_features = self.norm_edges(edge_features)

        return edge_features, neighbor_idx

class ProteinFeatures(nnx.Module):
    def __init__(self, edge_features, node_features, rngs: nnx.Rngs, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        kernel_init = jax.nn.initializers.glorot_uniform()
        """Extract full-atom protein edge features."""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings, rngs=rngs)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nnx.Linear(
            edge_in, edge_features, use_bias=False, rngs=rngs, kernel_init=kernel_init
        )
        self.norm_edges = nnx.LayerNorm(edge_features, rngs=rngs)

    def _dist(self, ca_coords, mask, eps=1e-6):
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = ca_coords[:, None, :, :] - ca_coords[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, ca_coords.shape[1])
        neighbor_idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        neighbor_distances = jnp.take_along_axis(D_adjust, neighbor_idx, axis=-1)
        return neighbor_distances, neighbor_idx

    def _rbf(self, distances):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count).reshape((1, 1, 1, -1))
        D_sigma = (D_max - D_min) / D_count
        D_expand = distances[..., None]
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, atom_coords_a, atom_coords_b, neighbor_idx):
        pairwise_distances = jnp.sqrt(jnp.sum((atom_coords_a[:, :, None, :] - atom_coords_b[:, None, :, :]) ** 2, axis=-1) + 1e-6)  # [B, L, L]
        neighbor_distances = gather_edges(pairwise_distances[..., None], neighbor_idx)[..., 0]  # [B, L, K]
        rbf_features = self._rbf(neighbor_distances)
        return rbf_features

    def __call__(self, atom_coords, mask, residue_idx, chain_labels, noise_key=None):
        if self.augment_eps > 0 and noise_key is not None:
            atom_coords = atom_coords + self.augment_eps * jax.random.normal(noise_key, shape=atom_coords.shape, dtype=atom_coords.dtype)

        b = atom_coords[:, :, 1, :] - atom_coords[:, :, 0, :]
        c = atom_coords[:, :, 2, :] - atom_coords[:, :, 1, :]
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + atom_coords[:, :, 1, :]
        Ca = atom_coords[:, :, 1, :]
        N = atom_coords[:, :, 0, :]
        C = atom_coords[:, :, 2, :]
        O = atom_coords[:, :, 3, :]

        neighbor_distances, neighbor_idx = self._dist(Ca, mask)

        rbf_feature_blocks = []
        rbf_feature_blocks.append(self._rbf(neighbor_distances))      # Ca-Ca
        rbf_feature_blocks.append(self._get_rbf(N, N, neighbor_idx))  # N-N
        rbf_feature_blocks.append(self._get_rbf(C, C, neighbor_idx))  # C-C
        rbf_feature_blocks.append(self._get_rbf(O, O, neighbor_idx))  # O-O
        rbf_feature_blocks.append(self._get_rbf(Cb, Cb, neighbor_idx))  # Cb-Cb
        rbf_feature_blocks.append(self._get_rbf(Ca, N, neighbor_idx))  # Ca-N
        rbf_feature_blocks.append(self._get_rbf(Ca, C, neighbor_idx))  # Ca-C
        rbf_feature_blocks.append(self._get_rbf(Ca, O, neighbor_idx))  # Ca-O
        rbf_feature_blocks.append(self._get_rbf(Ca, Cb, neighbor_idx))  # Ca-Cb
        rbf_feature_blocks.append(self._get_rbf(N, C, neighbor_idx))   # N-C
        rbf_feature_blocks.append(self._get_rbf(N, O, neighbor_idx))   # N-O
        rbf_feature_blocks.append(self._get_rbf(N, Cb, neighbor_idx))  # N-Cb
        rbf_feature_blocks.append(self._get_rbf(Cb, C, neighbor_idx))  # Cb-C
        rbf_feature_blocks.append(self._get_rbf(Cb, O, neighbor_idx))  # Cb-O
        rbf_feature_blocks.append(self._get_rbf(O, C, neighbor_idx))   # O-C
        rbf_feature_blocks.append(self._get_rbf(N, Ca, neighbor_idx))  # N-Ca
        rbf_feature_blocks.append(self._get_rbf(C, Ca, neighbor_idx))  # C-Ca
        rbf_feature_blocks.append(self._get_rbf(O, Ca, neighbor_idx))  # O-Ca
        rbf_feature_blocks.append(self._get_rbf(Cb, Ca, neighbor_idx))  # Cb-Ca
        rbf_feature_blocks.append(self._get_rbf(C, N, neighbor_idx))   # C-N
        rbf_feature_blocks.append(self._get_rbf(O, N, neighbor_idx))   # O-N
        rbf_feature_blocks.append(self._get_rbf(Cb, N, neighbor_idx))  # Cb-N
        rbf_feature_blocks.append(self._get_rbf(C, Cb, neighbor_idx))  # C-Cb
        rbf_feature_blocks.append(self._get_rbf(O, Cb, neighbor_idx))  # O-Cb
        rbf_feature_blocks.append(self._get_rbf(C, O, neighbor_idx))   # C-O
        rbf_all = jnp.concatenate(rbf_feature_blocks, axis=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], neighbor_idx)[..., 0]  # [B, L, K]

        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        edge_chain_mask = gather_edges(d_chains[:, :, :, None], neighbor_idx)[..., 0]
        edge_positional_features = self.embeddings(offset.astype(jnp.int32), edge_chain_mask)
        edge_features = jnp.concatenate((edge_positional_features, rbf_all), axis=-1)
        edge_features = self.edge_embedding(edge_features)
        edge_features = self.norm_edges(edge_features)
        return edge_features, neighbor_idx
