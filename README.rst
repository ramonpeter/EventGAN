===============================================
Generative Adversarial Networks for LHC Events
===============================================

This is a **Tensorflow 2** implementation to GAN [1]_ LHC events.

* Use of a **LorentzVector** layer to implement on-shell conditions.
* Adds JS-Regularizer to the discriminator objective (Roth et al, 2017).
* Adds Maximum mean discrepancy (MMD) to capture resonances.


Paper
--------------

This repository contains the code to reproduce the results shown in the paper:

- **"How to GAN LHC Events" (2019)** `SciPost Phys. 7, 075 (2019)`_, `1907.03764 [hep-ph]`_.
  
.. _`SciPost Phys. 7, 075 (2019)` : https://scipost.org/10.21468/SciPostPhys.7.6.075
.. _`1907.03764 [hep-ph]`: https://arxiv.org/abs/1907.03764

A more detailed explaination has been given at 

- `ML4 Jets 2020`_.

.. _`ML4 Jets 2020`: https://indico.cern.ch/event/809820/contributions/3632585/attachments/1970203/3278531/GAN_LHC.pdf

Installation
-------------

Dependencies
~~~~~~~~~~~~

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | >= 3.7                        |
+---------------------------+-------------------------------+
| Tensorflow                | >= 2.1.0                      |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.15.0                     |
+---------------------------+-------------------------------+


Download + Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: sh

   # clone the repository
   git clone https://github.com/itp-mlhep/EventGAN.git
   # then run the main file with the param_card
   cd EventGAN
   python event_gan cards/PARAM_CARD.yaml


Citation
---------

If you use this code, please cite:

.. code:: sh

    @article{Butter:2019cae,
        author = "Butter, Anja and Plehn, Tilman and Winterhalder, Ramon",
        title = "{How to GAN LHC Events}",
        eprint = "1907.03764",
        archivePrefix = "arXiv",
        primaryClass = "hep-ph",
        doi = "10.21468/SciPostPhys.7.6.075",
        journal = "SciPost Phys.",
        volume = "7",
        number = "6",
        pages = "075",
        year = "2019"
    }


.. [1] From ‘to GAN’, in close analogy to the verbs taylor, google, and sommerfeld.
