import matplotlib.pyplot as plt
import numpy as np

def check_image():
    try:
        img = plt.imread('results/shadow_cat_result.png')
        print(f"Image Shape: {img.shape}")
        print(f"Min: {img.min():.4f}, Max: {img.max():.4f}")
        print(f"Mean: {img.mean():.4f}, Std: {img.std():.4f}")
        
        if img.std() < 0.01:
            print("WARNING: Image has very low contrast (effectively blank).")
        else:
            print("Image has valid contrast.")
            
    except Exception as e:
        print(f"Error reading image: {e}")

if __name__ == "__main__":
    check_image()
