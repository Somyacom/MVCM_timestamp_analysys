import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import os

class MVCMGapFiller:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.filled_data = None
        self.missing_patterns = {}


    def load_from_csv(self, file_path, delimiter=','):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        df = pd.read_csv(file_path, delimiter=delimiter)
        self.data = df.values
        self.original_data = self.data.copy()
        print(f"Loaded data with shape: {self.data.shape}")
        return self
    

    def load_from_array(self, data_array):
        self.data = np.array(data_array)
        self.original_data = self.data.copy()
        print(f"Loaded data with shape: {self.data.shape}")
        return self
    

    def create_missing_patterns(self, patterns=None):
        if patterns is None:
            n_points = self.data.shape[0]
            self.missing_patterns = {
                0: np.arange(20, 30),
                1: np.arange(40, 50),
                2: np.arange(60, 70),
                3: np.arange(80, 90)
            }
        else:
            self.missing_patterns = patterns
        data_with_gaps = self.data.copy()
        for var_idx, missing_indices in self.missing_patterns.items():
            if var_idx < self.data.shape[1]:
                data_with_gaps[missing_indices, var_idx] = np.nan
        
        self.data = data_with_gaps
        return self


    def generate_test_data(self, n_points=100, n_vars=4):
        def generate_correlated_vectors(n, a=0.8, noise_level=0.2):
            X = np.random.rand(n)
            Z = np.random.rand(n) * noise_level
            Y = a * X + Z
            return X, Y
        
        data_vectors = []
        for i in range(n_vars // 2):
            X, Y = generate_correlated_vectors(n_points)
            data_vectors.extend([X, Y])
        while len(data_vectors) < n_vars:
            data_vectors.append(np.random.rand(n_points))
        
        self.data = np.vstack(data_vectors).T
        self.original_data = self.data.copy()
        return self
    

    def smap_weights(self, distances, theta):
        if len(distances) == 0:
            return np.array([])
        denom = np.mean(distances) if np.mean(distances) > 0 else 1.0
        weights = np.exp(-theta * distances / denom)
        weights /= np.sum(weights)
        return weights


    def smap_cross_map(self, embedded_data, target, query_embedding, theta=1.0, n_neighbors=3):
        if len(embedded_data) == 0 or np.isnan(query_embedding).any():
            return np.nan
        min_len = min(len(target), len(embedded_data))
        if min_len == 0:
            return np.nan

        target = target[:min_len]
        embedded_data = embedded_data[:min_len]

        valid_target = ~np.isnan(target)
        valid_embedded = ~np.isnan(embedded_data).any(axis=1)
        valid_mask = valid_target & valid_embedded

        if np.sum(valid_mask) == 0:
            return np.nan

        emb = embedded_data[valid_mask]
        tgt = target[valid_mask]

        if len(emb) == 0:
            return np.nan

        n_neighbors = min(n_neighbors, len(emb))
        if n_neighbors == 0:
            return np.nan

        try:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(emb)
            distances, indices = nbrs.kneighbors(query_embedding.reshape(1, -1))
        except Exception as e:
            return np.nan

        distances = distances.flatten()
        indices = indices.flatten()

        if len(distances) == 0:
            return np.nan

        weights = self.smap_weights(distances, theta)
        if len(weights) == 0:
            return np.nan

        X_neighbors = emb[indices]
        y_neighbors = tgt[indices]

        try:
            reg = LinearRegression()
            reg.fit(X_neighbors, y_neighbors, sample_weight=weights)
            pred = reg.predict(query_embedding.reshape(1, -1))
            return pred[0]
        except:
            return np.nan


    def create_embedding(self, timeseries, lags, embedding_dim):
        n_samples = len(timeseries) - (embedding_dim - 1) * lags
        if n_samples <= 0:
            return np.array([])
        emb = []
        for start in range(n_samples):
            window = [timeseries[start + i*lags] for i in range(embedding_dim)]
            if np.any(np.isnan(window)):
                continue
            emb.append(window)
        return np.array(emb)


    def create_multivariate_embedding(self, data, view, lags, embedding_dim):
        emb_list = []
        for v in view:
            emb = self.create_embedding(data[:, v], lags, embedding_dim)
            if len(emb) == 0:
                return np.array([])
            emb_list.append(emb)

        if not emb_list:
            return np.array([])

        lengths = [emb.shape[0] for emb in emb_list]
        min_len = min(lengths)
        emb_list = [emb[:min_len] for emb in emb_list]
        emb = np.hstack(emb_list)
        return emb


    def mvcm_gap_fill(self, target_idx, lags=1, embedding_dim=2, theta=1.0,
                      n_neighbors=2, top_n_views=3, verbose=True):
        
        data = self.data.copy()
        n_vars = data.shape[1]
        target = data[:, target_idx]
        missing_idx = np.where(np.isnan(target))[0]

        if len(missing_idx) == 0:
            if verbose:
                print(f"No missing values found for variable {target_idx}")
            return data

        var_indices = [i for i in range(n_vars) if i != target_idx]
        if len(var_indices) < embedding_dim:
            if verbose:
                print(f"Not enough variables for embedding_dim={embedding_dim}")
            return data

        all_views = list(combinations(var_indices, embedding_dim))

        max_start = len(target) - (embedding_dim - 1) * lags
        if max_start <= 0:
            if verbose:
                print("Time series too short for embedding parameters.")
            return data

        view_skills = []
        for view in all_views:
            emb = self.create_multivariate_embedding(data, view, lags, embedding_dim)
            if len(emb) == 0:
                view_skills.append((view, -np.inf))
                continue

            tgt_emb = target[(embedding_dim - 1) * lags:]

            min_len = min(len(tgt_emb), emb.shape[0])
            if min_len == 0:
                view_skills.append((view, -np.inf))
                continue

            emb_sub = emb[:min_len]
            tgt_sub = tgt_emb[:min_len]

            valid_points = ~np.isnan(tgt_sub) & ~np.isnan(emb_sub).any(axis=1)
            if np.sum(valid_points) < embedding_dim + 2:
                view_skills.append((view, -np.inf))
                continue

            emb_sub = emb_sub[valid_points]
            tgt_sub = tgt_sub[valid_points]

            if len(emb_sub) < 2:
                view_skills.append((view, -np.inf))
                continue

            pred = []
            for i in range(len(tgt_sub)):
                q_emb = emb_sub[i]
                idxs = list(range(len(tgt_sub)))
                idxs.remove(i)
                if len(idxs) == 0:
                    pred.append(np.nan)
                    continue

                val = self.smap_cross_map(emb_sub[idxs], tgt_sub[idxs], q_emb, theta, n_neighbors)
                pred.append(val)

            pred = np.array(pred)
            valid_pred = ~np.isnan(pred)
            if np.sum(valid_pred) < 2:
                skill = -np.inf
            else:
                try:
                    skill = np.corrcoef(pred[valid_pred], tgt_sub[valid_pred])[0, 1]
                    if np.isnan(skill):
                        skill = -np.inf
                except:
                    skill = -np.inf

            view_skills.append((view, skill))

        view_skills = [vs for vs in view_skills if vs[1] != -np.inf]
        if len(view_skills) == 0:
            if verbose:
                print(f"No valid embedding views found for variable {target_idx}.")
            return data

        view_skills.sort(key=lambda x: -abs(x[1]))
        top_views = view_skills[:min(top_n_views, len(view_skills))]
        top_views = [v for v, s in top_views]
        top_skills = [abs(s) for v, s in view_skills[:min(top_n_views, len(view_skills))]]

        if len(top_skills) == 0 or np.sum(top_skills) == 0:
            weights = np.ones(len(top_views)) / len(top_views)
        else:
            weights = np.array(top_skills) / np.sum(top_skills)

        for idx in missing_idx:
            estimates = []
            for view in top_views:
                emb = self.create_multivariate_embedding(data, view, lags, embedding_dim)
                if len(emb) == 0:
                    estimates.append(np.nan)
                    continue

                emb_idx = idx - (embedding_dim - 1) * lags
                if emb_idx < 0 or emb_idx >= emb.shape[0]:
                    estimates.append(np.nan)
                    continue

                query_embedding = emb[emb_idx]
                tgt_emb = target[(embedding_dim - 1) * lags:]

                min_len = min(len(tgt_emb), emb.shape[0])
                if min_len == 0:
                    estimates.append(np.nan)
                    continue

                emb_val = emb[:min_len]
                tgt_emb_val = tgt_emb[:min_len]

                val = self.smap_cross_map(emb_val, tgt_emb_val, query_embedding, theta, n_neighbors)
                estimates.append(val)

            estimates = np.array(estimates)
            valid_estimates = ~np.isnan(estimates)

            if np.any(valid_estimates):
                filled_value = np.dot(estimates[valid_estimates], weights[valid_estimates]) / np.sum(weights[valid_estimates])
                data[idx, target_idx] = filled_value
            else:
                valid_target = target[~np.isnan(target)]
                if len(valid_target) > 0:
                    data[idx, target_idx] = np.mean(valid_target)

        return data

    def fill_all_gaps(self, **kwargs):
        self.filled_data = self.data.copy()
        results = {}
        
        for var_idx in range(self.data.shape[1]):
            missing_indices = np.where(np.isnan(self.data[:, var_idx]))[0]
            if len(missing_indices) > 0:
                print(f"Filling gaps for variable {var_idx}...")
                filled = self.mvcm_gap_fill(var_idx, **kwargs)
                self.filled_data[:, var_idx] = filled[:, var_idx]
                
                if hasattr(self, 'original_data'):
                    mse = mean_squared_error(
                        self.original_data[missing_indices, var_idx],
                        filled[missing_indices, var_idx]
                    )
                    results[var_idx] = {'mse': mse, 'missing_count': len(missing_indices)}
        
        return results


    def plot_comparison(self, var_names=None):
        if var_names is None:
            var_names = [f'Variable {i}' for i in range(self.data.shape[1])]
        
        n_vars = self.data.shape[1]
        fig, axes = plt.subplots(n_vars, 2, figsize=(15, 3*n_vars))
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        t = np.arange(len(self.data))
        
        for i in range(n_vars):
            axes[i, 0].plot(t, self.data[:, i], 'bo-', alpha=0.7, label='With gaps', markersize=3)
            axes[i, 0].set_title(f'{var_names[i]} - Original with gaps')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(t, self.filled_data[:, i], 'r-', label='Filled', linewidth=2)
            if hasattr(self, 'original_data'):
                axes[i, 1].plot(t, self.original_data[:, i], 'g--', alpha=0.7, label='True')
            axes[i, 1].set_title(f'{var_names[i]} - After gap filling')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        corr_before = np.corrcoef(self.data, rowvar=False)
        im1 = ax1.imshow(corr_before, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Correlation before gap filling')
        plt.colorbar(im1, ax=ax1)
        
        corr_after = np.corrcoef(self.filled_data, rowvar=False)
        im2 = ax2.imshow(corr_after, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Correlation after gap filling')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

    def save_results(self, output_path):
        if self.filled_data is not None:
            df = pd.DataFrame(self.filled_data)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

def main():
    filler = MVCMGapFiller()
    
    # Вариант 1: Генерация тестовых данных
    filler.generate_test_data(n_points=100, n_vars=4)
    filler.create_missing_patterns()
    
    # Вариант 2: Загрузка из файла
    # filler.load_from_csv('your_data.csv').create_missing_patterns()
    
    results = filler.fill_all_gaps(
        lags=1,
        embedding_dim=2,
        theta=1.0,
        n_neighbors=2,
        top_n_views=3
    )
    
    print("\n=== Results Summary ===")
    for var_idx, result in results.items():
        print(f"Variable {var_idx}: MSE = {result['mse']:.5f}, Missing points = {result['missing_count']}")
    
    var_names = ["Variable 0", "Variable 1", "Variable n", "Variable z"]
    filler.plot_comparison(var_names)
    filler.plot_correlation_matrix()
    
    filler.save_results('filled_data.csv')

if __name__ == "__main__":
    main()