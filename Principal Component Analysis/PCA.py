from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):

    face_data = np.load(filename)
    centered_face_data = face_data - np.mean(face_data, axis=0)
    return centered_face_data.astype(np.float64)

def get_covariance(dataset):

    n = dataset.shape[0]
    return 1/(n-1)*np.dot(np.transpose(dataset), dataset)

def get_eig(S, k):

    #eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0] - k, S.shape[0] - 1])
    #sort descending
    sorted_desc = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)

    eigenvalues = np.array(eigenvalues)[sorted_desc]
    eigenvectors = eigenvectors[:, sorted_desc]
    
    #diagonal matrix of k-largest eigenvalues, corresponding eigenvectors
    return np.diag(eigenvalues[:k]), eigenvectors[:, :k]

def get_eig_prop(S, prop): 
    #get eig for all eigenvalues
    eigenvalues, eigenvectors = eigh(S)
    
    sorted_desc = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    eigenvalues = np.array(eigenvalues)[sorted_desc]
    eigenvectors = eigenvectors[:, sorted_desc]
    
    #proportion of variance explained by each eigenvalue
    var_prop = eigenvalues / np.sum(eigenvalues)

    # filter eignenvalues and eigenvectors based on prop
    signif_index = np.where(var_prop >= prop)[0]
    
    # filter eigenvalues and eigenvectors based on signif_index
    eigenvalues = eigenvalues[signif_index]
    eigenvectors = eigenvectors[:, signif_index]

    # return diagonal matrix of eigenvalues, corresponding eigenvectors
    return np.diag(eigenvalues), eigenvectors

def project_image(image, U): 

    # calculate the projection coefficients of the image onto the eigenvectors
    proj_coeff = np.dot(np.transpose(U), image)  
    
    # return reconstructed image from the projection coefficients
    return np.dot(U, proj_coeff) 
    
def display_image(orig, proj): 
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2

    # reshape images to 64x64
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)

    # settings for image 1
    ax1.set_title("Original")
    img1 = ax1.imshow(orig_image, aspect='equal', cmap='viridis')
    plt.colorbar(img1, ax=ax1)

    # settings for image 2
    ax2.set_title("Projection")
    img2 = ax2.imshow(proj_image, aspect='equal', cmap='viridis')
    plt.colorbar(img2, ax=ax2)

    return fig, ax1, ax2

def perturb_image(image, U, sigma):

    # calculate the projection coefficients of the image onto the eigenvectors
    proj_coeff = np.dot(np.transpose(U), image) 
    
    perturbed = np.random.normal(0, sigma, size=proj_coeff.shape)
    
    # Return reconstructed image from the perturbed projection coefficients
    return np.dot(U, proj_coeff + perturbed)

def combine_image(image1, image2, U, lam):

    # Calculate the projection coefficients of the two images onto the eigenvectors
    img1_coeff = np.dot(np.transpose(U), image1)  
    img2_coeff = np.dot(np.transpose(U), image2)  
    
    combination = lam * img1_coeff + (1 - lam) * img2_coeff 
    
    # Return the combined image from the combined projection coefficients
    return np.dot(U, combination)


def display_image_combo(image1, image2, combined_image):
    # Step 1: Reshape the images to 64x64
    image1_reshaped = image1.reshape(64, 64)
    image2_reshaped = image2.reshape(64, 64)
    combined_image_reshaped = combined_image.reshape(64, 64)

    # Step 2: Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3)

    # Step 3: Display the original images and the combined image
    ax1.set_title("Image 1")
    img1 = ax1.imshow(image1_reshaped, aspect='equal', cmap='viridis')
    plt.colorbar(img1, ax=ax1)

    ax2.set_title("Image 2")
    img2 = ax2.imshow(image2_reshaped, aspect='equal', cmap='viridis')
    plt.colorbar(img2, ax=ax2)

    ax3.set_title("Combined Image")
    img_comb = ax3.imshow(combined_image_reshaped, aspect='equal', cmap='viridis')
    plt.colorbar(img_comb, ax=ax3)

    # Step 4: Return figure and axis objects
    return fig, ax1, ax2, ax3


def display_image_perturbed(orig, perturbed_image):
    # Step 1: Reshape the original and perturbed images to 64x64
    orig_image = orig.reshape(64, 64)
    perturbed_image_reshaped = perturbed_image.reshape(64, 64)

    # Step 2: Create a figure with one row of two subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)

    # Step 3: Title the first subplot as "Original" and the second as "Perturbed"
    ax1.set_title("Original")
    ax2.set_title("Perturbed")

    # Step 4: Use imshow to display the images, with aspect='equal'
    img1 = ax1.imshow(orig_image, aspect='equal', cmap='viridis')
    img2 = ax2.imshow(perturbed_image_reshaped, aspect='equal', cmap='viridis')

    # Step 5: Create a colorbar for each image
    plt.colorbar(img1, ax=ax1)
    plt.colorbar(img2, ax=ax2)

    # Step 6: Return the figure and axes objects
    return fig, ax1, ax2


# Test functions
def test_projection(dataset):
    S = get_covariance(dataset)
    Lambda, U = get_eig(S, 100)
    projection = project_image(dataset[50], U)
    fig, ax1, ax2 = display_image(dataset[50], projection)
    plt.show()

def test_perturbation(dataset):
    S = get_covariance(dataset)
    Lambda, U = get_eig(S, 100)
    perturbed_image = perturb_image(dataset[50], U, 1000)
    fig, ax1, ax2 = display_image_perturbed(dataset[50], perturbed_image)
    plt.show()

def test_combination(dataset):
    S = get_covariance(dataset)
    Lambda, U = get_eig(S, 100)
    combined_image = combine_image(dataset[50], dataset[80], U, 0.5)
    fig, ax1, ax2, ax3 = display_image_combo(dataset[50], dataset[80], combined_image)
    plt.show()

def main():
    # Load the dataset
    dataset = load_and_center_dataset('face_dataset.npy')
    
    # Run all tests
    print("Running Projection Test:")
    test_projection(dataset)

    print("Running Perturbation Test:")
    test_perturbation(dataset)

    print("Running Combination Test:")
    test_combination(dataset)

if __name__ == "__main__":
    main()




