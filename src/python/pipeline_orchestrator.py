import subprocess

def run_split_data(base_filename, num_files):
    subprocess.run(['python', 'split_data.py', base_filename, str(num_files)])

def run_train_validate():
    subprocess.run(['python', 'train_validate.py'])

def run_test_model():
    subprocess.run(['python', 'test_model.py'])

def main(base_filename, num_files):
    run_split_data(base_filename, num_files)
    run_train_validate()
    run_test_model()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orchestrate the entire pipeline.')
    parser.add_argument('data', type=str, help='The base name of the CSV files')
    parser.add_argument('num_files', type=int, help='The number of CSV files')
    args = parser.parse_args()

    main(args.data, args.num_files)