""" LHE Writer. """


# pylint: disable=C0103, R0903
class LHEWriter:
    """
    This class creates a writer to output the events generated by the GAN
    in the Les Houches Event file format.
    """

    def __init__(self, file: str, n_particles: int):

        self.file = open(file, "w")
        self.n_particles = n_particles

    # ================
    # PRIVATE METHODS
    # ================

    def _write_intro(self):
        """ Writes the header """

        self.file.write("<header>\n")

        header = """
        #################################################################################
        #                      THIS FILE HAS BEEN PRODUCED BY                           #
        #===============================================================================#
        #                                                                               #
        #  ********                             **     ********      **     ****     ** #
        # /**/////                             /**    **//////**    ****   /**/**   /** #
        # /**       **    **  *****  *******  ****** **      //    **//**  /**//**  /** #
        # /******* /**   /** **///**//**///**///**/ /**           **  //** /** //** /** #
        # /**////  //** /** /******* /**  /**  /**  /**    ***** **********/**  //**/** #
        # /**       //****  /**////  /**  /**  /**  //**  ////**/**//////**/**   //**** #
        # /********  //**   //****** ***  /**  //**  //******** /**     /**/**    //*** #
        # ////////    //     ////// ///   //    //    ////////  //      // //      ///  #
        #                                                                               #
        #                                                                               #
        #            Authors: Anja Butter, Tilman Plehn, Ramon Winterhalder             #
        #                                                                               #
        #   "How to GAN LHC Events", SciPost Phys. 7, 075 (2019), 1907.03764 [hep-ph]   #
        #                                                                               #
        #                                                                               #
        #################################################################################
        #################################################################################
        #   The original Les Houches Event (LHE) format is defined in hep-ph/0609017    #
        #################################################################################

        """

        self.file.write(header)
        self.file.write("</header>\n")

        # here there are supposed to be general information on the process
        self.file.write("<init>\n")
        self.file.write("</init>\n")

    def _write_particle(self, momentum, mass, pdg):
        """ Writes a single particle """

        E, px, py, pz = momentum

        self.file.write(
            "      %2i    1    0    0    0    0  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  0.00000  0.00000\n"
            % (pdg, px, py, pz, E, mass)
        )

    def _write_event(self, event, masses, pdg_codes):
        """ Writes a single event"""

        self.file.write("<event>\n")

        self.file.write(
            "%i   0  -1.0000000E+00 -1.0000000E+00 -1.0000000E+00 -1.0000000E+00\n"
            % (self.n_particles)
        )

        for momentum, mass, pdg in zip(event, masses, pdg_codes):

            self._write_particle(momentum, mass, pdg)

        self.file.write("</event>\n")

    # ================
    # PUBLIC METHODS
    # ================

    def write_lhe(self, events, masses, pdg_codes):
        """
        This function takes as inputs:
        - events: 2D array containing events in rows. Each row is an array
                  of 4 momenta, one after the other.
        - masses: list with masses for the particles, in the same order
                  the momenta are given.
        - pdg_codes: list with pdg codes for the particles, in the same
                     order the momenta are given.

        Output:
        A lhe file containing all the events.
        """

        if len(pdg_codes) != self.n_particles:
            raise Exception(
                f"You have passed {len(pdg_codes)} pdg codes for {self.n_particles} particles."
            )

        self.file.write('<LesHouchesEvents version="3.0">\n')

        self._write_intro()

        for event in events:
            event = event.reshape(self.n_particles, 4)

            self._write_event(event, masses, pdg_codes)

        self.file.write("</LesHouchesEvents>\n")
        self.file.close()
