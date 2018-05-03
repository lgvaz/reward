.. role:: hidden
    :class: hidden-section

Models
======

.. automodule:: torchrl.models
.. currentmodule:: torchrl.models

BaseModel
---------

.. autoclass:: BaseModel
  :show-inheritance:
  :members:
  :private-members:

ValueModel
----------

.. autoclass:: ValueModel
  :show-inheritance:
  :exclude-members: add_losses, train
  :members:
  :private-members:

BasePGModel
-----------

.. autoclass:: BasePGModel
  :show-inheritance:
  :exclude-members: add_losses, train
  :members:
  :private-members:

VanillaPGModel
--------------

.. autoclass:: VanillaPGModel
  :show-inheritance:
  :exclude-members: add_losses, train
  :members:
  :private-members:

SurrogatePGModel
----------------

.. autoclass:: SurrogatePGModel
  :show-inheritance:
  :exclude-members: add_losses, train
  :members:
  :private-members:
    
PPOModel
--------

.. autoclass:: PPOModel
  :show-inheritance:
  :exclude-members: add_losses, train
  :members:
  :private-members:
    
