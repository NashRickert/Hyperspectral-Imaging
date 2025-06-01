// Redo comments for modified functions
// In general, clean up the code
// Add clean up for all my buffers (especially for non-inplace transfer of data -- too much memory in use)
// Remove excess comments
// Determine if the Conv3d macros are appropriate or will overlap with other things we are interested in

// Modify function calls to take pointers to structs to save on the copying overhead. Structs have gotten large enough this is probably a good idea
