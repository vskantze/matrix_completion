import argparse
from data.data_loader import DataHandler
from utils.matrix_completion import MatrixHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("data_path ", default=r"C:\Users\ViktorSkantze\OneDrive - Fraunhofer-Chalmers Centre\Python\projects\Pose estimation\data\kaggle_data\imgs\photos1\Im_L_1.png")
    # Add more arguments as needed
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    data_handler = DataHandler(data_path=args.data_path)
    df_data = data_handler.load_data()

    # Do matrix completion
    matrix_handler = MatrixHandler(matrix=df_data)
    matrix_handler.sparsify_matrix()
    matrix_handler.matrix_completion()





if __name__ == '__main__':
    main()