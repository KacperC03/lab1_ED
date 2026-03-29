from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movies = {id : Movie.index[id] for id in Movie.index}
    def jaccard_similarity(self, movie_a: Movie, movie_b: Movie) -> float:
        genres_a = set(movie_a.genres)
        genres_b = set(movie_b.genres)
        if not genres_a and not genres_b:
            return 1.0
        union = genres_a | genres_b
        return len(genres_a & genres_b) / len(union)
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
        # user_avg=self.calc_user_avg(user)
        num=0
        movie_avg=0
        for mv in self.movies:
            jacc_sim = self.jaccard_similarity(Movie.index[movie],Movie.index[mv])
            if(jacc_sim==1.0):
                movie_avg+=self.calc_movie_avg(mv)
                num+=1
        movie_avg/=num
        if num==0:
            return self.calc_movie_avg(movie)
        return movie_avg

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 156007 and 155833'
