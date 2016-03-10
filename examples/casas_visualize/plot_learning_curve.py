import os
import sys
from actlearn.data.casas import plot_casas_learning_curve


if __name__ == '__main__':
    # Check if pkl files are provided
    if len(sys.argv) <= 1:
        sys.stderr.write('Please provide list of pkl files that contains learning curve data.\n')
    # Check if all pkl files exists, if they do, append them to lists
    learning_curve_files = []
    for i in range(1, len(sys.argv)):
        if os.path.exists(str(sys.argv[i])):
            learning_curve_files.append(str(sys.argv[i]))
        else:
            sys.stderr.write('Cannot find file %s\n' % str(sys.argv[i]))
    if len(learning_curve_files) == 0:
        sys.stderr.write('No learning curve files found\n')
    else:
        plot_casas_learning_curve(learning_curve_files)

