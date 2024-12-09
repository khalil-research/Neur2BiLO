from argparse import ArgumentParser

from blo.data_manager import factory_dm


#-----------------------------------------------------------------------#
#                                                                       #
#                       File to generate datasets                       #
#                                                                       #
#-----------------------------------------------------------------------#


def main(args):

    # get data manager.
    data_manager = factory_dm(args.problem)

    # run commands
    data_manager.generate_dataset(args.n_procs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp")
    parser.add_argument('--n_procs', type=int, default=1)
    args = parser.parse_args()
    main(args)
