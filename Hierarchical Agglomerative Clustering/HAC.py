import csv
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    #read csv file path
    with open (filepath, 'r') as file:
        read = csv.DictReader(file)
        data = [row for row in read]
    return data

def calc_features(row):
    #calculate the feature vector
    vector = [row['Population'], row['Net migration'], row['GDP ($ per capita)'], row['Literacy (%)'], 
              row['Phones (per 1000)'], row['Infant mortality (per 1000 births)']]
    
    #convert to float64
    vector = np.array([np.float64(x) for x in vector])
    return vector

#helper function to create distance matrix
def create_distance_matrix(features):
    n = len(features)
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = np.linalg.norm(features[i] - features[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix


def hac(features):
    # initialize n to length of features
    n = len(features)
    # initialize cluster indices
    cluster_indices = [[i] for i in range(n)]
    # initialize output array
    output = np.zeros((n - 1, 4))
    # initialize distance matrix
    distance_matrix = create_distance_matrix(features)

    for i in range(n - 1):  # iterate through the number of clusters
        # for each new iteration, reset the minimum distance and candidate pairs
        min_dist = np.inf
        candidate_pairs = []  # store pairs with the same minimum distance

        # iterate through the cluster indices
        for j in range(len(cluster_indices)):
            # skip merged clusters
            if cluster_indices[j] is None:
                continue
            for k in range(j + 1, len(cluster_indices)):
                if cluster_indices[k] is None:
                    continue

                # find min dist between clusters
                current_min_dist = np.min([distance_matrix[c1, c2] for c1 in cluster_indices[j] for c2 in cluster_indices[k]])

                # if new minimum distance is found, reset candidate pairs
                if current_min_dist < min_dist:
                    min_dist = current_min_dist
                    candidate_pairs = [(j, k)]
                # if the current distance equals the minimum, add the pair to the candidates
                elif current_min_dist == min_dist:
                    candidate_pairs.append((j, k))

        # tie-breaking clusters--sort pairs by first index, then by second index
        first_cluster, second_cluster = min(candidate_pairs, key=lambda pair: (pair[0], pair[1]))

        # merge the two closest clusters
        cluster_indices.append(cluster_indices[first_cluster] + cluster_indices[second_cluster])
        #mark clusters as merged
        cluster_indices[first_cluster] = None  
        cluster_indices[second_cluster] = None 

        # add the merge in the output array 
        output[i, 0] = first_cluster
        output[i, 1] = second_cluster
        output[i, 2] = min_dist
        output[i, 3] = len(cluster_indices[-1])  # number of countries in the new cluster

    return output


def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation = 90)
    plt.tight_layout()
    plt.show()
    return fig

def normalize_features(features):
    features = np.array(features) #convert to numpy array
    normalized = np.zeros(features.shape) #initialize normalized array

    #normalize each feature
    for i in range(len(features)):
        for j in range(6):
            col_min = np.min(features[:, j])
            col_max = np.max(features[:, j])

            if col_min == col_max: #check for zero division case 
                normalized[i][j] = 0 
            else:
                normalized[i][j] = (features[i][j] - col_min) / (col_max - col_min)

    #convert to list 
    normalized_list = [normalized[i] for i in range(len(normalized))]

    return normalized_list


# Test functions
def test_hac(filepath):
    data = load_data(filepath)
    
    # Ask the user to input the number of rows to run HAC on
    n = int(input(f"Enter the number of rows to run HAC on (max {len(data)}): "))
    
    # Check if n is valid
    if n > len(data) or n <= 0:
        print(f"Invalid number. Please enter a number between 1 and {len(data)}.")
        return

    # Use only the first n rows
    data_subset = data[:n]

    # Calculate and normalize the features
    features = [calc_features(row) for row in data_subset]
    normalized_features = normalize_features(features)

    # Perform HAC clustering
    Z = hac(normalized_features)

    # Extract names for the dendrogram labels
    names = [row['Country'] for row in data_subset]

    # Plot the dendrogram
    fig_hac(Z, names)

def main():
    # Specify the CSV file path
    csv_file = 'socioeconomic_data.csv'
    
    # Test the HAC algorithm and dendrogram visualization
    print("Running HAC Test:")
    test_hac(csv_file)

if __name__ == "__main__":
    main()



#Load the data
# data = load_data('countries.csv')
# country_names = [row['Country'] for row in data]
# features = [calc_features(row) for row in data]

# # Normalize features using all countries
# features_normalized = normalize_features(features)

# # Run HAC on the first 50 normalized countries
# n = 204
# Z_normalized = hac(features_normalized[:n])

# # Plot the dendrogram for the first 50 countries using normalized data
# fig = fig_hac(Z_normalized, country_names[:n])
# plt.show()

# # ---------------------------------------------------------------------------------- 
# # Compare with the provided output.txt file

# # Load the output.txt file
# output_file = 'output.txt'
# Z_output = np.loadtxt(output_file)

# # Compare with your generated Z_normalized
# comparison = np.allclose(Z_normalized, Z_output, atol=1e-6)  # Check if matrices are close
# if comparison:
#     print("Your HAC output matches the provided output.txt!")
# else:
#     print("Your HAC output does not match output.txt. Check your implementation.")

# # You can also plot the dendrogram for comparison if needed
# fig = fig_hac(Z_normalized, country_names[:50])
# plt.show()  # This will plot the hierarchical clustering dendrogram