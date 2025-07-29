## Overview:

For the sake of collecting the outputs for the lookup tables, it is necessary to make a modified version of the .forward function inside of KANLayer.py in the KAN library. The only difference is that we do not do y = torch.sum(y, dim=1). This is because that gives us the summed output, but we want to retain the outputs for each activation before summing. The easiest way to do this is to simply copy the function inside of the library, modify it by removing that line, and rename it to something usable (such as act_forward or perhaps something more expressive).

Note also that early on I had an issue that required a very minor modification to the library as well. I do not remember what it was or if others might get the same issue. If they do, it should not be hard to see it based on inspecting the error message and associated library function. It had to do with an uninvertible matrix and was resolvable by uncommenting (or moving out of a control block) some code in the library that applied a small amount of regularization.

Note that right now I have no garbage collection for dynamic C memory. I claim this is ok becuase most memory needs to survive the lifetime of the function anyways (except tensors) (although usage of ffi.gc makes me think these might already be getting garbage collected? Because using it gives me an error, so it might be trying to double free).

The forward function with a table size of 4096 takes 0.00085-0.00087 seconds to run in C (from the start of the forward function to the end) for a single sample on my CPU. I ran the test 5 times and all of them fell in this range. Note that the model this is tested on has dimensions [200, 32, 32, 32, 16]. This is with the following CPU specs: CPU: 12th Gen Intel(R) Core(TM) i7-1260P (16) @ 4.70 GHz.


## Usage Instructions:

Currently build_kan.py is used to build the C library, kan_test.py is used to actually use the library and perform forward computation, and test_kan.py is used purely for testing. Thus python build_kan.py && python kan_test.py is sufficient for running it currently. The library requirements are included in the requirements.txt in the upper level of this directory.

The way the code works is to first construct and then run forward inference on an existing python KAN model through the usage of lookup tables. We use CFFI API mode to interface with the C code. Documentation can be found here https://cffi.readthedocs.io/en/latest/using.html (pay particular attention to the discussion of ownership and lifetimes -- this is important to understand).

The C code currently contians a macro called SCALE which determines whether we scale (aka normalize) the output of each lookup table to [0,1]. The only change to remove this is commenting #define SCALE 1 in load.h -- the python code will automatically adjust for this. We are able to expose simple C macros to python through global C variables accessed through our library. CFFI is not perfectly compatible with all forms of macros -- refer to the documentation.

I recommend observing the structs inside of load.h to get a sense of how the code works. Essentially we have a parent struct model which contains a buffer of layers. Each layer in turn contains a buffer of nodes that compose the layer. Each node has an adder tree which stores inputs to the node from previous layers which will need to be added. It also had activation functions for every node in the next layer it is connected to as well as a pointer to the next layer. Each activation function has a lookup table which contains the lookup values and some meta data. Reading my final report will also give more insight on this.

Code that intializes/loads/or provides general utility is in load.c, code for actually performing the forward inference is in forward.c. In kan_test.py we use the information from our loaded KAN model in python to do our full initializtion (including calculating the proper values for each lookup table), then do a forward pass by callign lib.forward. This is useful because a lot of non-inference operations, including tensor operations, can take place in python where they're easier.

Example usage: On any stable commit, unless I change something, running the code should do an example, including showing the difference between the python inference and the C inference (differences attributable to float precision + operation differences and the fact that we approximate with lookup tables).

