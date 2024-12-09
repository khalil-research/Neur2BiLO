from argparse import ArgumentParser

from blo.data_manager import factory_dm

#-----------------------------------------------------------------------#
#                                                                       #
#                       File to generate problem                        #  
#                                                                       #
#-----------------------------------------------------------------------#


def main(args):

    # get data manager.
    data_manager = factory_dm(args.problem)

    # generate problem
    data_manager.initialize_problem()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp")
    args = parser.parse_args()
    main(args)
