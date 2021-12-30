###################################
# main.py
###################################
# Description:
# * Pull data, train model (or download 
# pretrained model) and generate
# neural transferred image.

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

from argparse import ArgumentParser

def main():
    """
    * Perform key steps in order.
    """
    args = get_args()
    input_image, transfer_image = get_data(args)
    input_image, transfer_image = transform_data(args)
    model = get_model(args)

def get_args():
    """
    * Get command line arguments.
    """
    parser = ArgumentParser('neural_transfer')
    parser.add_argument('input_image')
    parser.add_argument('transfer_image')
    parser.add_argument('--download')


