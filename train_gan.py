""" Main trainings script """

# ********                             **     ********      **     ****     **
# /**/////                             /**    **//////**    ****   /**/**   /**
# /**       **    **  *****  *******  ****** **      //    **//**  /**//**  /**
# /******* /**   /** **///**//**///**///**/ /**           **  //** /** //** /**
# /**////  //** /** /******* /**  /**  /**  /**    ***** **********/**  //**/**
# /**       //****  /**////  /**  /**  /**  //**  ////**/**//////**/**   //****
# /********  //**   //****** ***  /**  //**  //******** /**     /**/**    //***
# ////////    //     ////// ///   //    //    ////////  //      // //      ///

# EnventGAN - generative adversarial network based event generator for HEP.
# Copyright (C) 2021 Ramon Winterhalder

import time
import os
import argparse

import yaml

from eventgan.model import EventGAN
from eventgan.utils.lhe_writer import LHEWriter
from eventgan.utils.plots import plot_loss

# pylint: disable=C0103
if __name__ == "__main__":

    ########################################
    # Parse YAML files
    ########################################

    print("Parse YAML files...")

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType("r"))
    args = parser.parse_args()

    with args.file as f1:
        param_args = yaml.load(f1, Loader=yaml.FullLoader)

    ########################################
    # Configuration
    ########################################

    default_params = {
        # Train the models
        "training": True,
        "save_weights": False,
        "plot_losses": False,
        "use_mmd_loss": True,
        # intermediate action
        "save_intermediate_weights": True,
        "load_intermediate_weights": False,
        "save_epochs": [100, 200, 300, 400, 600],
        "load_epochs": [100, 200, 300, 400, 600],
        # Event generation
        "n_events": 1000000,
        "save_lhe": True,
        # Input/Output/Name
        "save_path": "outputs",
        "train_data_path": "datasets/ttbar/ttbar_6f_train.h5",
        "test_data_path": "datasets/ttbar/ttbar_6f_test.h5",
        "scaler": 450.0,
        "input_masses": [0.0, 0.0, 4.7, 0.0, 0.0, 4.7],
        "input_pdgs": [2, -1, 5, -2, 1, -5],
        "run_tag": "paper_01",
        # Training parameters
        "batch_size": 1024,
        "iterations_per_epoch": 1000,
        "epochs": 1000,
        "train_updates_d": 1,
        "train_updates_g": 1,
        "train_fraction": 1.0,
        # Optimizer configurations
        "optimizer_args": {
            "g_lr": 0.001,
            "g_beta_1": 0.5,
            "g_beta_2": 0.9,
            "g_decay": 0.1,
            "d_lr": 0.001,
            "d_beta_1": 0.5,
            "d_beta_2": 0.9,
            "d_decay": 0.1,
        },
        # loss weights
        "loss_weights": {"reg_weight": 0.001, "mmd_weight": 1.0},
        # Process specific input
        "mmd_kernel": "breit-wigner-mix",
        "mmd_kernel_widths": [(1.49,), (1.49,), (2.05,), (2.05,)],
        "topology": [(0, 1), (3, 4), (0, 1, 2), (3, 4, 5)],
        # Parameters for model architectures
        "latent_dim": 18,
        "n_particles": 6,
        "g_units": 512,
        "d_units": 512,
        "g_layers": 10,
        "d_layers": 10,
    }

    #####################################################################
    # Read in parameters
    #####################################################################

    print("Read in parameters..")

    params = {}
    for param in default_params:

        if param in param_args.keys():
            cls = default_params[param].__class__
            value = cls(param_args[param])
            params[param] = value
        else:
            params[param] = default_params[param]

    # Make output dir
    output_dir = params["save_path"]
    file_path = str(params["epochs"]) + "epochs"
    file_path += "/" + params["run_tag"]
    log_dir = os.path.abspath(os.path.join(output_dir, file_path))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if params["save_intermediate_weights"]:
        log_dir_inter = log_dir + "/intermediate"
        if not os.path.exists(log_dir_inter):
            os.makedirs(log_dir_inter)

    ###########################################################################
    # Build Model
    ###########################################################################

    print("Build model..")

    gan_params = {
        "n_particles": params["n_particles"],
        "latent_dim": params["latent_dim"],
        "topology": params["topology"],
        "input_masses": params["input_masses"],
        "train_data_path": params["train_data_path"],
        "train_updates_d": params["train_updates_d"],
        "train_updates_g": params["train_updates_g"],
        "train_fraction": params["train_fraction"],
        "test_data_path": params["test_data_path"],
        "scaler": params["scaler"],
        "g_units": params["g_units"],
        "g_layers": params["g_layers"],
        "d_units": params["d_units"],
        "d_layers": params["d_layers"],
        "reg_weight": params["loss_weights"]["reg_weight"],
        "use_mmd_loss": params["use_mmd_loss"],
        "mmd_weight": params["loss_weights"]["mmd_weight"],
        "mmd_kernel": params["mmd_kernel"],
        "mmd_kernel_widths": params["mmd_kernel_widths"],
    }

    gan = EventGAN(**gan_params)

    #######################################################################
    # Training of the GAN
    #######################################################################

    if params["training"]:
        train_params = {
            "optimizer_args": params["optimizer_args"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "iterations": params["iterations_per_epoch"],
            "safe_weights": params["save_intermediate_weights"],
            "safe_epochs": params["save_epochs"],
            "log_dir": log_dir,
        }

        start_time = time.time()
        print("Start training...")
        c_loss, g_loss = gan.train(**train_params)
        print("--- Run time: %s hour ---" % ((time.time() - start_time) / 60 / 60))
        print("--- Run time: %s mins ---" % ((time.time() - start_time) / 60))
        print("--- Run time: %s secs ---" % ((time.time() - start_time)))

        # Plot the losses
        print("Save loss plots...")

        if params["plot_losses"]:
            plot_loss(
                c_loss[params["iterations_per_epoch"] - 1 :: params["iterations_per_epoch"]],
                name="C",
                log_dir=log_dir,
            )

            plot_loss(
                g_loss[params["iterations_per_epoch"] - 1 :: params["iterations_per_epoch"]],
                name="G",
                log_dir=log_dir,
            )
    else:
        print("Load weights..")
        gan.load_weights(log_dir)

    #######################################################################
    # Generate events and save as LHE file
    #######################################################################

    if params["save_lhe"]:
        print("Save LHE event file...")
        n_events = params["n_events"]
        events = gan.get_events(params["n_events"])
        lhe = LHEWriter(log_dir + "/events.lhe", params["n_particles"])
        lhe.write_lhe(events, params["input_masses"], params["input_pdgs"])

    ######################################################
    # Save the weights
    ######################################################

    if params["save_weights"] and params["training"]:
        print("Save weights..")
        gan.save_weights(log_dir)

    print("Execution finished")
