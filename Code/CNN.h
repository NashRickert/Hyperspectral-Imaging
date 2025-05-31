// Redo comments for modified functions
// In general, clean up the code
// Do prefix caching for get_idx
// Add malloc checks since some allocations are quite large
// Add clean up for all my buffers (especially for non-inplace transfer of data -- too much memory in use)
// Remove excess comments
// Determine if the Conv3d macros are appropriate or will overlap with other things we are interested in
// Also probably not a bad idea to cache data/weight length to avoid recomputation, even though it isn't that intensive. Just better to avoid get_size calls
