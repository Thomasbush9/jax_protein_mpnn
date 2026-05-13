import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops


#===== Gather functions

def gather_edges(edges, neighbor_idx):
    # features [B, N, N , C] n_idx [B, N, K] -> neighbor features [B, N, K, C]

    idx = neighbor_idx[..., None]                        # (B, N, K, 1)
    idx = jnp.broadcast_to(idx, (*neighbor_idx.shape, edges.shape[-1]))  # (B, N, K, C)
    edge_features = jnp.take_along_axis(edges, idx, axis=2)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # nodes: [B, N, C], neighbor_idx: [B, N, K] -> [B, N, K, C]
    b, n, k = neighbor_idx.shape
    c = nodes.shape[-1]
    idx = einops.rearrange(neighbor_idx, 'b n k -> b (n k)')   # [B, NK]
    idx = idx[..., None]                                        # [B, NK, 1]
    idx = jnp.broadcast_to(idx, (b, n * k, c))                 # [B, NK, C]
    neighbor_features = jnp.take_along_axis(nodes, idx, axis=1) # [B, NK, C]
    neighbor_features = einops.rearrange(
        neighbor_features, 'b (n k) c -> b n k c', n=n, k=k
    )
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # features [B, N, C] a N_idx [B, K] -> neighbor_features [B, K, C]
    b, n, c = nodes.shape
    _, k = neighbor_idx.shape
    idx_flat = neighbor_idx[..., None]
    idx_flat = jnp.broadcast_to(idx_flat, (b, k, c))
    neighbor_features = jnp.take_along_axis(nodes, idx_flat, axis=1)
    return neighbor_features


#=== Position Layers
class PositionalEncodings(nnx.Module):
    def __init__(self, num_embeddings, rngs: nnx.Rngs, max_relative_feature=32)-> None:
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nnx.Linear(2*max_relative_feature+1+1, num_embeddings, rngs=rngs)

    def __call__(self, offset, mask):
        d = jnp.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature) *mask
        d_onehot = jax.nn.one_hot(d, num_classes=2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot)
        return E

class CA_ProteinFeatures(nnx.Module):
    def __init__(self, edge_features, node_features, rngs:nnx.Rngs, num_positional_embeddings=16 ,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16)-> None:

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
        self.node_embedding = nnx.Linear(node_in, node_features, use_bias=False, rngs=rngs)
        self.edge_embedding = nnx.Linear(edge_in, edge_features, use_bias=False, rngs=rngs)
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

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # X: [B, N, 3], E_idx: [B, N, K]
        dX = X[:, 1:, :] - X[:, :-1, :]                         # [B, N-1, 3]
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
        O_neighbors = gather_nodes(O, E_idx)          # [B, N, K, 9]
        X_neighbors = gather_nodes(X, E_idx)          # [B, N, K, 3]
        O = O.reshape(*O.shape[:2], 3, 3)             # [B, N, 3, 3]
        O_neighbors = O_neighbors.reshape(*O_neighbors.shape[:3], 3, 3)  # [B, N, K, 3, 3]
        # Relative orientation
        dX = X_neighbors - X[:, :, None, :]           # [B, N, K, 3]
        dU = jnp.matmul(O[:, :, None, :, :], dX[..., None]).squeeze(-1)   # [B, N, K, 3]
        dU = self._normalize(dU, axis=-1)
        R = jnp.matmul(jnp.swapaxes(O[:, :, None, :, :], -1, -2), O_neighbors)  # [B,N,K,3,3]
        Q = self._quaternions(R)                      # [B, N, K, 4]
        O_features = jnp.concatenate([dU, Q], axis=-1)  # [B, N, K, 7]
        return AD_features, O_features
    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = X[:, None, :, :] - X[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)

        # Identify k nearest neighbors (including self)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1. - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        E_idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        D_neighbors = jnp.take_along_axis(D_adjust, E_idx, axis=-1)
        mask_neighbors = gather_edges(mask_2D[..., None], E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape((1, 1, 1, -1))
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(jnp.sum((A[:, :, None, :] - B[:, None, :, :])**2, axis=-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[..., None], E_idx)[..., 0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, Ca, mask, residue_idx, chain_labels, noise_key=None):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0 and noise_key is not None:
            Ca = Ca + self.augment_eps * jax.random.normal(noise_key, shape=Ca.shape, dtype=Ca.dtype)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = jnp.zeros_like(Ca)
        Ca_2 = jnp.zeros_like(Ca)
        Ca_0 = Ca_0.at[:, 1:, :].set(Ca[:, :-1, :])
        Ca_1 = Ca
        Ca_2 = Ca_2.at[:, :-1, :].set(Ca[:, 1:, :])

        V, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))


        RBF_all = jnp.concatenate(RBF_all, axis=-1)


        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0] #[B, L, K]

        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
        E = jnp.concatenate((E_positional, RBF_all, O_features), axis=-1)


        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx

class ProteinFeatures(nnx.Module):
    def __init__(self, edge_features, node_features, rngs: nnx.Rngs, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """Extract full-atom protein edge features."""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings, rngs=rngs)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nnx.Linear(edge_in, edge_features, use_bias=False, rngs=rngs)
        self.norm_edges = nnx.LayerNorm(edge_features, rngs=rngs)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = X[:, None, :, :] - X[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        E_idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        D_neighbors = jnp.take_along_axis(D_adjust, E_idx, axis=-1)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count).reshape((1, 1, 1, -1))
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[..., None], E_idx)[..., 0]  # [B, L, K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, X, mask, residue_idx, chain_labels, noise_key=None):
        if self.augment_eps > 0 and noise_key is not None:
            X = X + self.augment_eps * jax.random.normal(noise_key, shape=X.shape, dtype=X.dtype)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))      # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))   # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))   # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))   # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))   # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))   # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))   # C-O
        RBF_all = jnp.concatenate(RBF_all, axis=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[..., 0]  # [B, L, K]

        d_chains = (chain_labels[:, :, None] == chain_labels[:, None, :]).astype(jnp.int32)
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[..., 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
        E = jnp.concatenate((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx
