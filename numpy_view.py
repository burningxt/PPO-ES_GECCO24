import numpy as np
import argparse

def load_and_print_npy(file_path):
    try:
        # Set NumPy print options to avoid truncation
        np.set_printoptions(threshold=np.inf)
        
        # Load the .npy file
        data = np.load(file_path)
        
        # Print the contents of the file
        print(data)
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load and print a .npy file")
    parser.add_argument("file_path", help="Path to the .npy file")
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Load and print the .npy file
    load_and_print_npy(args.file_path)

if __name__ == "__main__":
    main()
