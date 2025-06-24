from _kan import lib, ffi
import torch
from kan import *
import torch
from kan.spline import coef2curve
import weakref

torch.manual_seed(7)
torch.cuda.manual_seed(7)

width = [200, 32, 32, 32, 16]
grid = 10
k = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This function modifies the width variable for some reason >:(. So we redeclare right after I guess
# The other variables are thankfully unmodified
model = KAN(width=width, grid = grid, k=k, seed=42, device=device, symbolic_enabled=False, auto_save=False)
width = [200, 32, 32, 32, 16]

model.load_state_dict(torch.load('temp_model.pt', map_location=device))
        
# Done to sync C macro def with this code. In the C code, we assigned a global to have the macro val so we can do this
TBL_SIZE = lib.TABLE_SIZE
SCALE = lib.SCALED

# Create the C object for the model
c_model = ffi.new("struct model *")
c_model = lib.init_model(width, len(width))

# Accepts something like model.act_fun[i], a layer of input
# Computes the evenly spaced input that will be used to construct the lookup tables
def compute_layer_input(layer):
    grid = layer.grid
    x = []

    # For each input dim (aka for each node)
    for i in range(len(grid)):
        minimum = grid[i, 0]
        maximum = grid[i, -1]
        eval_pts = torch.linspace(start=minimum, end=maximum, steps=TBL_SIZE)
        x.append(eval_pts)
    x = torch.stack(x)
    return (x.T).to(device)

# Given a layer, constructs the outputs of the lookup tables associated with the layer
def compute_layer_output(layer):
    x = compute_layer_input(layer)
    with torch.no_grad():
        y, _, _, _ = layer.act_forward(x)
    return y.to(device)

# Finds the meta info that is associated with each lookup table
# By meta info, we mean xmin, xmax, xdist, and inv_xdist as in the C code
def get_meta_info(layer):
    x = compute_layer_input(layer)
    mins = torch.min(x, dim=0).values
    maxs = torch.max(x, dim=0).values
    xdists = (maxs - mins) / TBL_SIZE
    inv_xdists = (1 / xdists)

    meta_info = torch.cat((mins, maxs, xdists, inv_xdists))
    return meta_info
    

global_weakkeydict = weakref.WeakKeyDictionary()

def make_tens(dimensions, data=None):
    # Construct the shape and add to dict
    c_shape = ffi.new("struct Shape *")
    c_shape.len = len(dimensions)
    # Note: Due to lifetimes it is actually necessary to do this pattern instead of assigning directly
    dim = ffi.new("int[]", dimensions)
    c_shape.dim = dim

    # Construct the tensor and add to dict
    # We allocate all fields inside the C code, so no concern about lifetimes yet
    c_tens = ffi.new("struct Tensor *")
    c_tens[0] = lib.construct_tensor(c_shape[0])

    if data is None:
        return c_tens

    # We are now concerned about lifetimes because of how we assign the field
    c_data = ffi.new("float[]", data)
    c_tens.data = c_data

    global_weakkeydict[c_tens] = c_data

    return c_tens


    
# This loop is used to instantiate the lookup tables for the model by looping through each layer
for i, func in enumerate(model.act_fun):
    # Compute the table values
    y = compute_layer_output(func)

    c_meta_tens = make_tens([width[i], 4], get_meta_info(func).tolist())

    # We are not scaling
    if SCALE == 0:
        # This is a nice pattern to turn a python tensor to a c tensor
        c_val_tens = make_tens(list(y.shape), torch.flatten(y).tolist())

        lib.fill_lkup_tables(c_val_tens, c_meta_tens, ffi.NULL, ffi.addressof(c_model.layers[i]))
        continue
        
    y_maxes = torch.max(y, dim=0).values
    y_mins = torch.min(y, dim=0).values

    y_diff = y_maxes - y_mins

    # y and y_diff/min shape mismatch doesn't matter here because of broadcasting
    z = (y - y_mins) / y_diff

    c_val_tens = make_tens(list(y.shape), torch.flatten(z).tolist())

    y_meta_data = torch.cat((y_mins.flatten(), y_maxes.flatten()))

    c_mm_tens = make_tens([2, func.in_dim, func.out_dim], y_meta_data.tolist())

    # Now that our tensors are constructed, we call the proper C function on the layer
    lib.fill_lkup_tables(c_val_tens, c_meta_tens, c_mm_tens, ffi.addressof(c_model.layers[i]))

    
# Utility function (which actually isn't that useful) for printing model info
def print_cmodel_info(c_model):
    for i in range(len(width) - 1):
        layer = c_model.layers[i]
        print(f"In layer {i} with length {layer.len} and self index {layer.idx}")
        for j in range(layer.len):
            nodes = layer.nodes[j]
            print(f"In the {j}th node with value {nodes.val}, target length {nodes.len}")
            table = nodes.funcs[0].table
            print(f"Our xmax, xmin, xdist, and inv_xdist respectively are {table.xmin}, {table.xmax}, {table.xdist}, {table.inv_xdist}")
            for k in range(nodes.len):
                print(f"Our targets are {nodes.targets[k]}")
            
x = torch.rand((1,200)).to(device)

with torch.no_grad():
    y = model(x).to(device)

# instantiate C input parameters and the return parameters
c_input = ffi.new("float[]", x.flatten().tolist())
length = 200
c_retbuf = ffi.new("float **")
c_retlen = ffi.new("int *")
lib.forward(ffi.addressof(c_model), c_input, length, c_retbuf, c_retlen)

# Retrieve the return parameteres
out_len = c_retlen[0]
retbuf = c_retbuf[0]
result = []

# Put all the output results in a single list (from the C object)
for i in range(out_len):
    result.append(retbuf[i])

# Compare the diffs
print("Python model result:", y)
print("KAN C model result:", result)
