import csv
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILENAME = 'DataSet.csv'


def merge_csv():
    data_dir = os.path.join(CUR_DIR, 'GuardaCSV')
    with open(os.path.join(data_dir, DATA_FILENAME), 'w') as datafile:
        is_first_file = True
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and filename != DATA_FILENAME:
                with open(os.path.join(data_dir, filename)) as f:
                    header = f.readline()
                    if is_first_file:
                        datafile.write(header)
                    for line in f:
                        datafile.write(line)
                    is_first_file = False


if __name__ == '__main__':
    merge_csv()
