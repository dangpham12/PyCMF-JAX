# PyCMF with JAX

## Objectives

This project is based on previous works called **PyCMF** and **PyCMF-GT4Py** which were done by **Nathan Marotte** and **Matthias Van der Cam** respectively, which aimed to experiment Object-Oriented technology and the use of Python in a Climate Modelling context.
It concluded that OO brings a lot of flexibility and modularity to the code, but that it is not the best choice for performance. 
Moreover, the GT4Py improves drastically the performance while being portable for all architectures.
All visualisation implementations were abandoned.

The goal with this project is to use **JAX library** to provide a performant and flexible framework for climate modelling, while keeping the modularity and flexibility of the previous projects.
With this implementation, we aim to provide a framework that could be also compared to the GT4Py implementation enhanced.

## Introduction

Python Climate Modelling Framework (or PyCMF for short) is a framework developped by Nathan Marotte as part of his Master's thesis at the Université Libre de Bruxelles. 

- base_class (earth_base, sun_base, etc ...) and inheriting from BaseModel : Contains the basic
  structural/pythonic stuff for the class (correct inheritance, redefinition of dunder methods, etc ...)
- physical_class (earth, sun, etc ...) and inheriting from its base_class : Contains the physical properties (temperature, mass, etc ...) and
  method (behaviour for receiving electromagnetic radiation, etc ... ) for that class 
- ticking_class (ticking_earth, ticking_universe, etc ...) and inheriting from its physical_class as well as
  TickableModel, that is an interface to store all the class methods that have to be executed at each time step of the
  simulation via the `@TickableModel.on_tick` decorator

The framework is currently **not** able to provide accurate simulations of real-world physical process, but provides a
few examples with placeholder simulations such as the averaging of the temperature at each time step


PyCMF-GT4Py was a project that aimed to use the GT4Py library to generate stencils for the models. 
- Stencils methods were defined with @gtscript.stencil decorator,
- Practical functions only used in those stencils were defined with @gtscript.function decorator
- Parallelization was done with the `with computation(keyword), interval(...)`, horizontal plane is already parallelized but the interval could be determined. `keyword` could take the values of `PARALLEL`, `FORWARD`, `BACKWARD`.

## Structure of the code

Although JAX was created for machine learning, it is a powerful library that can be used for numerical computing and scientific computing in general.

We used those following features of JAX to implement the framework:
- JAX's `jit()` decorator to compile functions for performance.
- JAX's `vmap()` to vectorize functions over arrays, allowing for efficient batch processing.

We also took advantage of the full reimplementation of the NumPy API in JAX, which allows us to use some functions to apply arrays transformations. 
It should be noted that JAX arrays are immutable and multiple restrictions have to be respected to use correctly the JAX library.
## Running the code

### Required Libraries

- JAX library


To run the framework, you can edit the script in `main.py` and then execute it with `python3.11 src/main.py`.

## How to add a new model

To add a new model, you have to add between 1 and 3 files since the models are split in 3 different layers. First, create your $model.py file in physical_class and add all the physical properties of that model you need. Then, if necessary, create another file in base_class to handle all the pythonic behavior, such as iteration behavior, adding, substracting, memory use, data storage, saving the simulation, loading a simulation, etc ... Don't forget to make your second layer model inherit from the first layer model.

Finally, if you model has some variables that are updated over time, you will have to create a third file in ticking_class to define the different updates behavior with the `@TickingModel.on_tick(enabled=True)` decorator that you obtain by inheriting from TickingModel. Also, you must inherit from your second layer class to get all of your physical properties variables on which the update is done.

## How to add a new variable to the model

To enrich the framework, you can add new physical properties and all their associated methods for conveniance (getters, setters, etc ... ) in the second layer, physical_class. You can also change the `__init__` method of the class to allow for setting your variable when the model is built, in which case you will have to find the other use of that model and change the constructor's parameters as well.

Then, if your variable has a temporal dimension to it, you can add the temporal evolution in the third layer, ticking_class, where you can define a function decorated by `@TickingModel.on_tick(enabled=True)`


