from _kan import lib, ffi

model = ffi.new("struct model *")
model = lib.init_model([8,16,16,8], 4)
print(model.len)
for i in range(model.len):
    layer = ffi.new("struct layer *", model.layers[i])
    print("layer idx:" + str(layer.idx))
    print("layer len:" + str(layer.len))
