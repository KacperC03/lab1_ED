import numpy as np
from system156007 import MySystem
import itertools
from RatingLib import Movie, User
from tqdm import tqdm
import csv

def train_hyperparameters():
    tag_options = [5, 10, 15]
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4]
    threshold_options = [0.05, 0.1, 0.15, 0.2, 0.3]
    combinations = list(itertools.product(tag_options, weight_options, threshold_options))
    
    best_rmse = float('inf')
    best_params = {}

    print(f"Starting Grid Search on {len(combinations)} combinations...")
    print(f"{'Tags':<6} | {'Weight':<8} | {'Thresh':<8} | {'RMSE'}")
    print("-" * 45)

    for tags, w, t in combinations:
        system = MySystem(num_tags=tags, weight=w, threshold=t)
        try:
            avg_rmse = system.cross_validate(k=3, sample_size=5000)
            print(f"{tags:<6} | {w:<8} | {t:<8} | {avg_rmse:.4f}")
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = {'num_tags': tags, 'weight': w, 'threshold': t}
        except Exception as e:
            print(f"Error testing {tags}, {w}, {t}: {e}")

    print("\n" + "="*30)
    print("TRAINING COMPLETE")
    print(f"Best Parameters: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("="*30)
    
    return best_params
def main():
    #read the movie indices
    with open('./data/movie.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in csv_reader:
            Movie(int(line[0]), line[1],line[2].split('|'))
    #read the genome scores
    last_movie = None
    with open('./data/genome_scores.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=11709768):
            movie_id = int(line[0])
            if last_movie == movie_id:
                if movie_id in Movie.index:
                    Movie.index[movie_id].add_genome_score(int(line[1]), float(line[2]))
            elif last_movie:
                if movie_id in Movie.index:
                    Movie.index[last_movie].sort_tags()
            last_movie = movie_id
    #read the user indices
    with open('./data/rating.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=20000263):
            if not int(line[0]) in User.index.keys():
                User(int(line[0]))
            User.index[int(line[0])].add_rating(Movie.index[int(line[1])],float(line[2]))
    train_hyperparameters()
if __name__ == "__main__":
    main()
    