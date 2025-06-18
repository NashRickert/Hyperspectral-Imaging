from _kan import lib, ffi
import torch
from kan import *
import torch
from kan.spline import coef2curve

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
        # y = coef2curve(x, layer.grid, layer.coef, k)
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
    

# If I want to make this into a function (which I should) then I might have to try using a weak dict or something
# To store references. Otherwise I think it would be really challenging to keep everything in scope

# This loop is used to instantiate the lookup tables for the model by looping through each layer
for i, func in enumerate(model.act_fun):
    # Compute the table values
    y = compute_layer_output(func)
    y_flat = torch.flatten(y)
    
    # Construct our shape for the tensor
    c_val_shape = ffi.new("struct Shape *")
    c_val_shape.len = 3

    # Note: Due to lifetimes it is actually necessary to do this pattern instead of assigning directly
    # This pattern repeats elsewhere in the code. Refer to the cffi documentation
    dim1 = ffi.new("int[]", list(y.shape))
    c_val_shape.dim = dim1

    # Construct out tensor and put our values in it
    c_val_tens = ffi.new("struct Tensor *")
    c_val_tens[0] = lib.construct_tensor(c_val_shape[0])

    data1 = ffi.new("float []", y_flat.tolist())
    c_val_tens.data = data1

    # Repeats the process for our meta data
    c_meta_shape = ffi.new("struct Shape *")
    c_meta_shape.len = 2
    dim2 = ffi.new("int[]", [width[i], 4])
    c_meta_shape.dim = dim2

    c_meta_tens = ffi.new("struct Tensor *")
    c_meta_tens[0] = lib.construct_tensor(c_meta_shape[0])

    meta_info = get_meta_info(func)
    mi = meta_info.tolist()

    data2 = ffi.new("float []", meta_info.tolist())
    c_meta_tens.data = data2

    # Now that our tensors are constructed, we call the proper C function on the layer
    lib.fill_lkup_tables(c_val_tens, c_meta_tens, ffi.addressof(c_model.layers[i]))

    
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
# print(x)
# print(x.shape)
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
