.. gausstorch documentation master file, created by
   sphinx-quickstart on Sun Nov 30 18:25:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gausstorch documentation
========================

This package allows for the simulation of coupled gaussian bosonic modes, with PyTorch for automatic differentiation.

I have developped this package for my PhD thesis, on the subject of "`Quantum machine learning with bosonic modes`".
`Here is a link to the manuscript <https://theses.hal.science/tel-05383369>`_.

However this is a simplified and cleaned-up version of my PhD code, meant for future PhD students to build up on.
In order to check out my actual PhD code, `here is a link to the Zenodo repository <https://zenodo.org/records/15856611>`_.

There are some significant differences in implementation, and the previous code is scarcely documented.
Most notably, my PhD code implements learning on benchmark machine learning tasks, which is not included here to keep the code short and quickly understandable.

The main feature of `gausstorch` is the :py:class:`gausstorch.libs.qsyst.Qsyst` class, whose methods return or plot time evolutions of coupled gaussian modes.
Its initiation arguments are the system drive, detuning, coupling, and dissipation parameters.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Contact
=======

If you have any questions, feel free to send me an email at julien.dudas[(at)]gmail.com