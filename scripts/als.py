import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Carregar os dados de ratings e filmes
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

def train_svd(data, n_components=20):
    """
    Treina um modelo SVD nos dados fornecidos, com IDs remapeados para uma sequência compacta.
    
    Input:
    - data: pandas DataFrame com colunas 'userId', 'movieId' e 'rating'
    - n_components: número de fatores latentes a serem usados no modelo SVD
    
    Return:
    - user_factors: matriz de fatores latentes dos usuários
    - item_factors: matriz de fatores latentes dos filmes
    """
    
    # Remapear os IDs de usuários e filmes para serem sequenciais e compactos
    user_map = {old_id: new_id for new_id, old_id in enumerate(data['userId'].unique())}
    movie_map = {old_id: new_id for new_id, old_id in enumerate(data['movieId'].unique())}
    
    # Aplicar o mapeamento
    data['userId_mapped'] = data['userId'].map(user_map)
    data['movieId_mapped'] = data['movieId'].map(movie_map)
    
    # Criar a matriz esparsa de usuário x filme
    row_indices = data['userId_mapped'].values
    col_indices = data['movieId_mapped'].values
    ratings_values = data['rating'].values
    
    # Matriz esparsa (usuário x filme)
    user_movie_sparse = csr_matrix((ratings_values, (row_indices, col_indices)))
    
    # Treinar o modelo SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_movie_sparse)  # Fatores latentes dos usuários
    item_factors = svd.components_.T  # Transpor para obter os fatores latentes dos filmes
    
    # Exibir as dimensões das matrizes resultantes
    print(f"Fatores de usuários (dimensões): {user_factors.shape}")
    print(f"Fatores de filmes (dimensões): {item_factors.shape}")
    
    return user_factors, item_factors

if __name__ == '__main__':
    user_factors, item_factors = train_svd(ratings)
