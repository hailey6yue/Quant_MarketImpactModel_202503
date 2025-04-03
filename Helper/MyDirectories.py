import os
import tarfile

class MyDirectories:

    @staticmethod
    def getProjectRoot():
        # returns the project root directory
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def getDataDir():
        # returns the path to the data folder
        return os.path.join(MyDirectories.getProjectRoot(),'data')

    @staticmethod
    def getTradesDir():
        # the path to the trades folder
        trades_dir = os.path.join(MyDirectories.getDataDir(),'trades')
        # check if .tar file exists and extract
        for file in os.listdir(trades_dir):
            if file.endswith('tar.gz'):
                tar_path = os.path.join(trades_dir,file)
                extracted_folder = os.path.join(trades_dir,file.replace('.tar.gz',''))
                if not os.path.exists(extracted_folder):
                    with tarfile.open(tar_path,'r:gz') as tar:
                        tar.extractall(trades_dir)
        return trades_dir

    @staticmethod
    def getQuotesDir():
        # the path to the trades folder
        quotes_dir = os.path.join(MyDirectories.getDataDir(),'quotes')
        # check if .tar file exists and extract
        for file in os.listdir(quotes_dir):
            if file.endswith('tar.gz'):
                tar_path = os.path.join(quotes_dir, file)
                extracted_folder = os.path.join(quotes_dir, file.replace('.tar.gz', ''))
                if not os.path.exists(extracted_folder):
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(quotes_dir)
        return quotes_dir