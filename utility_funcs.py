import os
import argparse

def get_file_names(directory, extention):
    ''' List all files with given exstention in given directory.

    Args:
        directory (str): path to files to be listed
        extention (str): extention of listed files

    Returns:
        list: list of filenames with given extention in directory
    '''

    return list(filter(lambda x: x.endswith(extention), os.listdir(directory)))

def get_args(args):
    ''' Gets command line arguments stated in args dictionary.

    Args:
        args (dict): dictionary, where key is argument name and value is argument description
    
    Returns:
        Namespace: Python Namespace containing arguments given to script
    '''

    parser = argparse.ArgumentParser()
    
    for name, description in args.items():
        parser.add_argument(name=name, help=description)
    
    return parser.parse_args()