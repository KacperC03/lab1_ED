from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie
import copy

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movies = {id : Movie.index[id] for id in Movie.index}
    def calc_movie_avg(self,movie):
        n = len(self.movie_ratings[movie])
        if n == 0:
            return 2.5
        else:
            return sum(self.movie_ratings[movie])/n
    def calc_user_avg(self,user):
        n = len(user.ratings.values())
        if n == 0:
            return 2.5
        else:
            return sum(user.ratings.values())/n
    def rate(self, user, movie):
        """
        Ta metoda zwraca rating w skali 1-5. Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        movie_avg=self.calc_movie_avg(movie)
        # user_avg=self.calc_user_avg(user)
        rate_num = len(user.ratings)
        movies_rated_by_user = list(user.ratings.keys())
        
        random_sample = np.random.choice(movies_rated_by_user, 100, replace=False) if rate_num >= 100 else np.random.choice(movies_rated_by_user, rate_num, replace=False)
        offset = 0
        for movie in random_sample:
            curr_movie = self.calc_movie_avg(movie)
            offset += curr_movie-user.ratings[movie]
        offset/=100
        return movie_avg-offset

    def cross_validate(self, k=5, sample_size=5000):
        all_interactions = []
        for user_id, user_obj in self.users.items():
            for movie_id, rating in user_obj.ratings.items():
                all_interactions.append((user_id, movie_id, rating))
                
        sample_size = min(sample_size, len(all_interactions))
        np.random.seed(42) # unseed before turn in
        indices = np.random.choice(len(all_interactions), sample_size, replace=False)
        cv_sample = [all_interactions[i] for i in indices]
        
        fold_size = len(cv_sample) // k
        rmse_scores = []
        
        for fold in range(k):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold != k-1 else len(cv_sample)
            test_set = cv_sample[test_start:test_end]
            
            errors = []
            for user_id, movie_id, true_rating in test_set:
                user_copy = copy.deepcopy(self.users[user_id])
                del user_copy.ratings[movie_id]
                
                pred_rating = self.rate(user_copy, movie_id)
                
                if pred_rating is None or np.isnan(pred_rating):
                    pred_rating = 2.5 
                    
                errors.append((true_rating - pred_rating)**2)
                
            fold_rmse = np.sqrt(np.mean(errors))
            rmse_scores.append(fold_rmse)
            print(f"Fold {fold+1}/{k} - RMSE: {fold_rmse:.4f}")
            
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        print(f"Cross-validation: Avg RMSE = {mean_rmse:.4f} (+/- {std_rmse:.4f})\n")
        
        return mean_rmse

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 156007 and 155833'