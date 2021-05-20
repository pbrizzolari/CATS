#!/usr/bin/env python3.8
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017
import argparse
import sys
# Start your coding
from sklearn.metrics import accuracy_score


from sklearn import linear_model
import pandas as pd
from joblib import dump, load

# import the library you need here

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding
    # setting up the model
    csf = load(args.model_file)

    #getting the data
    data = pd.read_csv(args.input_file, sep="\t")
    #index = [852, 854, 855, 999, 1000, 1001, 1002, 1004, 1026, 2184, 2206, 2207, 2210, 2213, 2214, 849, 853, 2223,
    #         2219, 673, 857, 1003, 846, 851, 2211, 672, 842, 2021, 2209, 2220, 674, 695, 850, 2056, 848, 2224, 694,
    #         837, 858, 840, 856, 1015, 2212, 2221, 844, 2208, 861, 2076, 2078, 692, 843, 1022, 1027, 2026, 2065, 845,
    #         818, 839, 847, 1016, 1050, 841, 2079, 1091, 2027, 2040, 671, 679, 863, 2039, 1025, 2064, 669, 1672, 2055,
    #         676, 696, 1062, 2023, 1035, 1656, 693, 2049, 2068, 691, 834, 1032, 2024, 697, 2074, 814, 819, 1034, 2075,
    #         859, 1061, 1567, 2017, 1055, 2750]
    #print(data.iloc[index,4:].transpose())
    test = data.iloc[index,4:].transpose()
    y_pred = csf.predict(test)
    with open(args.output_file, "w") as outfile:
        print(f'"Sample"\t"Subgroup"', file=outfile)
        for i, (x, y) in enumerate(zip(test.index,y_pred)):
            if i == len(test.index)-1:
                print(f'"{x}"\t"{y}"', file=outfile, end="")
            else:
                print(f'"{x}"\t"{y}"', file=outfile)



    # suggested steps
    # Step 1: load the model from the model file
    # Step 2: apply the model to the input file to do the prediction
    # Step 3: write the prediction into the desinated output file

    # End your coding


if __name__ == '__main__':
    main()
