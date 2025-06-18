from _kan import lib, ffi
import torch
from kan import *
import torch
from kan.spline import coef2curve

width = [200, 32, 32, 32, 16]
grid = 10
k = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This function modifies the width variable for some reason >:(. So we redeclare right after I guess
model = KAN(width=width, grid = grid, k=k, seed=42, device=device, symbolic_enabled=False, auto_save=False)
width = [200, 32, 32, 32, 16]

model.load_state_dict(torch.load('temp_model.pt', map_location=device))
        
# Done to sync C macro def with this code (Macro became global var accessible here)
TBL_SIZE = lib.TABLE_SIZE

c_model = ffi.new("struct model *")
c_model = lib.init_model(width, len(width))

# Accepts something like model.act_fun[i], a layer of input
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

# Note that most of the output values are small, some of them are entirely zero. Is this the desired behavior?
# ... could I have just done this with layer.forward? May want to check that vals end up being the same
do_print = True
def compute_layer_output(layer):
    x = compute_layer_input(layer)
    with torch.no_grad():
        y = coef2curve(x, layer.grid, layer.coef, k)

        y, _, _, _ = layer.act_forward(x)


        # if do_print:
        #     print(f"x shape: {x.shape}\n")
        #     print(f"With coef2curve: \n {y} \n with shape (coef2curve)\n {y.shape}")
        # print("With coef2curve: ", y)
        # y2, p1, p2, p3 = layer.forward(x)
        # print("With .forward: ", y2)
        # if do_print:
        #     print(f"Other shapes from .forward: {p1.shape}, {p2.shape}, {p3.shape} \n")
        #     print(f"With .forward: \n {y2} \n with shape (.forward)\n {y2.shape}")
            # do_print = False
    return y.to(device)

def get_meta_info(layer):
    x = compute_layer_input(layer)
    mins = torch.min(x, dim=0).values
    maxs = torch.max(x, dim=0).values
    xdists = (maxs - mins) / TBL_SIZE
    inv_xdists = (1 / xdists)

    meta_info = torch.cat((mins, maxs, xdists, inv_xdists))
    return meta_info
    
# May have to look into doing ffi.gc for tensor allocations (construct tensor in combination with destroy tensor)
# If I want to make this into a function (which I should) then I might have to try using a weak dict or something
# To store references. Otherwise I think it would be really challenging to keep everything in scope
for i, func in enumerate(model.act_fun):
    y = compute_layer_output(func)
    do_print = False
    y_flat = torch.flatten(y)
    
    c_val_shape = ffi.new("struct Shape *")
    c_val_shape.len = 3  # also: = len(list(y.shape))
    # Note: Due to lifetimes it is actually necessary to do this pattern instead of assigning directly
    dim1 = ffi.new("int[]", list(y.shape))
    c_val_shape.dim = dim1

    c_val_tens = ffi.new("struct Tensor *")
    c_val_tens[0] = lib.construct_tensor(c_val_shape[0])

    data1 = ffi.new("float []", y_flat.tolist())
    c_val_tens.data = data1 # ffi.new("float []", y_flat.tolist())

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

    lib.fill_lkup_tables(c_val_tens, c_meta_tens, ffi.addressof(c_model.layers[i]))

    
def print_cmodel_info(c_model):
    # for i in range(c_model.len):
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
            
# x = torch.normal(mean=torch.zeros(200), std = torch.ones(200))
x = torch.rand((1,200)).to(device)
print(x)
print(x.shape)
with torch.no_grad():
    y = model(x).to(device)
# print_cmodel_info(c_model)

c_input = ffi.new("float[]", x.flatten().tolist())
length = 200
c_retbuf = ffi.new("float **")
c_retlen = ffi.new("int *")
lib.forward(ffi.addressof(c_model), c_input, length, c_retbuf, c_retlen)

out_len = c_retlen[0]
retbuf = c_retbuf[0]
print(out_len)
result = []

for i in range(out_len):
    result.append(retbuf[i])

print("Python model result:", y)
print("KAN C model result:", result)
