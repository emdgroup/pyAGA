import pickle
import world_builder

from transformation_finder import find_trafos

with open("correlation_matrix_10x10_letters_with_f.pickle", "rb") as file:
    correlation_matrix = pickle.load(file)
    
  
print("2x2 world trafos:", find_trafos(world_builder.two_by_two_coeff, 3),"\n")

print("banner trafos:", find_trafos(world_builder.banner_coeff, 3), "\n")

print("from input:", find_trafos(correlation_matrix, 3))