/*!
 * \file    array.c
 * \brief   Array toolbox
 * \author  Adrien Python
 * \version 1.0
 * \date    21.11.2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "array.h"


/**
 * \brief Increment an index from last to first axis.
 *
 * \param   inc         increment
 * \param   dim         index dimension
 * \param   size        axis sizes
 * \param   index       index to increment
 * \param   fixed_axis  fixed axis to skip (-1 mean no fixed axis)
 * \return  increment left when the max index is reached, 0 otherwise.
 */
static inline size_t inc_index(size_t inc, size_t dim, const size_t size[dim], size_t index[dim], ssize_t fixed_axis)
{
    (void) index;
    
    ssize_t d = dim - (((ssize_t)dim-1 == fixed_axis) ? 2 : 1);
    
    while ( inc != 0 && d >= 0 ) {
        size_t a = index[d] + inc;
        size_t m = a % size[d];
        inc = a / size[d];
        index[d] = m;
        
        d -= (fixed_axis == d-1) + 1;
    }
    return inc;
}

/**
 * Additional arguments needed by the easy_print_wrapper function.
 */
typedef struct {
    const size_t* size;
    size_t i;
    array_easy_print_callback print;
} easy_print_wrapper_args;

/**
 * The foreach_callback handler, wrapping the array_easy_print_callback function
 * call, of the array_easy_print function.
 */
static bool easy_print_wrapper(size_t dim, void* data, const size_t index[dim], void* args)
{
    (void) index;

    easy_print_wrapper_args* epw_args = args;
    epw_args->print(data);
    putchar(++epw_args->i % epw_args->size[dim-1] ? ' ' : '\n');
    
    return true;
}

/**
 * \brief Create or map an array.
 *
 * \param   dim         array/mapping dimension
 * \param   size        axis sizes
 * \param   v_size      array data size (e.g. sizeof(int) for int array)
 * \param   base_array  base array to map (NULL for array creation)
 * \param   offset      mapping offset of the current index
 * \return  created array/mapping
 */
static void* array_creator(size_t dim, const size_t size[dim], size_t v_size,
                           void* base_array, size_t offset)
{
    void** array = NULL;
    
    switch (dim) {
        case 0:
            break;
        case 1:
            array = base_array ? (char*)base_array + offset : malloc(size[0] * v_size);
            break;
        default:
            array = malloc(size[0] * sizeof(void*));
            for (size_t i = 0; i < size[0]; i++) {
                array[i] = array_creator(dim-1, size+1, v_size, base_array, offset);
                offset += v_size * size[1];
            }
            break;
    }
    
    return array;
}

/**
 * \brief Destroy an array/mapping.
 *
 * \param   dim     array/mapping dimension
 * \param   size    axis sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   array/mapping to destroy
 * \param   map     indicates whether array is a mapping or not
 * \return  void
 */
static void array_destroyer(size_t dim, const size_t size[dim], size_t v_size, void* array, bool map)
{
    switch (dim) {
        case 0:
            break;
        case 1:
            if ( ! map ) {
                free(array);
            }
            break;
        default:
            for (size_t i = 0; i < size[0]; i++) {
                array_destroyer(dim-1, size+1, v_size, ((void**)array)[i], map);
            }
            free(array);
            break;
    }
}

/**
 * \brief Go at index position in the array and return its pointer.
 *
 * \param   dim     array dimension
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   array to 'walk' in
 * \param   index   position to 'walk' to
 * \return  pointer to the specified index position in the array
 */
static inline void* array_go_at(size_t dim, size_t v_size, void* array, const size_t index[dim])
{
    void* sub_array = array;
    for (size_t d = 0; d < dim-1; d++) {
        sub_array = ((void**)sub_array)[index[d]];
    }
    return (char*)sub_array + index[dim-1] * v_size;
}

/**
 * Additional arguments needed by the roll_to_handler function.
 */
typedef struct {
    size_t to_dim;
    const size_t* size;
    size_t v_size;
    void* to;
    size_t* to_index;
    const ssize_t* shift;
} roll_to_handler_args;

/**
 * The foreach_callback handler of array_roll_to function. Shift a cell to its
 * new position.
 */
static bool roll_to_handler(size_t dim, void* data, const size_t index[dim], void* rth_args)
{
    roll_to_handler_args* args = rth_args;
    
    for (size_t axis = 0; axis < dim; axis++) {
        args->to_index[axis] = (index[axis] + args->size[axis] + args->shift[axis]) % args->size[axis];
    }
    array_write_at(args->to_dim, args->v_size, args->to, data, args->to_index);
    
    return true;
}

/**
 * \brief Iterate on each cell of an array (array_foreach core).
 *
 * \param   dim         array dimension
 * \param   cur_dim     dimension the function is currently dealing with
 * \param   v_size      array data size (e.g. sizeof(int) for int array)
 * \param   array       array to iterate
 * \param   index       current index
 * \param   fixed_axis  axises to set to a fixed index; so, with dim=3:
 *                       -# {-1, -1, -1}: iterate all {x,y,z} array indexes
 *                       -# {-1, 4, -1}: iterate array for all {x, 4, z} indexes
 *                       -# NULL: equivalent to {-1, -1, -1}
 * \param   callback    function to call for each array cell
 * \param   args        callback function additional arguments
 * \return  indicates whether the function is iterating or stopped.
 */
static bool array_iterator(size_t dim, size_t cur_dim, size_t const size[cur_dim], size_t v_size, void* array,
                           size_t index[dim], ssize_t fixed_axis[dim], array_foreach_callback fct, void* args)
{
    bool continue_iteration = cur_dim != 0;
    
    size_t next_dim = cur_dim - 1;
    size_t index_dim = dim - cur_dim;
    size_t* next_size = (size_t*)size+1;

    bool is_fixed = fixed_axis && fixed_axis[index_dim] >= 0;
    
    size_t i = is_fixed ? fixed_axis[index_dim] : 0;

    do {
        index[index_dim] = i;
        if (cur_dim == 1) {
            continue_iteration = fct(dim, (char*)array + i * v_size, index, args);
        } else {
            continue_iteration = array_iterator(dim, next_dim, next_size, v_size, ((void**)array)[i], index, fixed_axis, fct, args);
        }
    } while (!is_fixed && continue_iteration && ++i < size[0]);

    return continue_iteration;
}

/**
 * The foreach_callback handler of array_set function. Set the cell's data.
 */
static bool set_handler(size_t dim, void* data, const size_t index[dim], void* args)
{
    (void) index;

    size_t* v_size = ((void**)args)[0];
    void* data_to_set = ((void**)args)[1];
    memcpy(data, data_to_set, *v_size);
    return true;
}

/**
 * Additional arguments needed by the axis_roll_handler function.
 */
typedef struct {
    const size_t size;
    size_t v_size;
    void* array;
    size_t idx_init;
    ssize_t shift;
    size_t axis;
} axis_roll_handler_args;

/**
 * The foreach_callback handler of array_roll_axis function. Roll (maybe 
 * partialy) a given axis.
 */
static bool axis_roll_handler(size_t dim, void* data, const size_t index[dim], void* arh_args)
{
    axis_roll_handler_args* args = arh_args;

    char tmp_mem[2 * args->v_size]; // static memory allocation for tmp
    void** tmp = (void**)tmp_mem;
    
    size_t i = 0;
    memcpy(&tmp[1], data, args->v_size);
    
    size_t idx1, idx0 = args->idx_init;
    
    size_t next_index[dim];
    memcpy(next_index, index, sizeof(next_index));
    
    do {
        // Compute next cell index
        next_index[args->axis] = idx1 = (idx0 + args->size + args->shift) % args->size;
        
        // Save its data
        void* data = array_go_at(dim, args->v_size, args->array, next_index);
        memcpy(&tmp[i % 2], data, args->v_size);
        
        // Shift the previous cell to the next
        memcpy(data, &tmp[(i+1) % 2], args->v_size);
        
        i++;
    } while ((idx0 = idx1) != args->idx_init);
    
    return true;
}

void* array_create(size_t dim, const size_t size[dim], size_t v_size)
{
    return array_creator(dim, size, v_size, NULL, 0);
}

void array_destroy(size_t dim, const size_t size[dim], size_t v_size, void* array)
{
    array_destroyer(dim, size, v_size, array, false);
}

void* array_map(size_t dim, const size_t size[dim], size_t v_size, void* array)
{
    return array_creator(dim, size, v_size, array, 0);
}

void array_unmap(size_t dim, const size_t size[dim], size_t v_size, void* array)
{
    array_destroyer(dim, size, v_size, array, true);
}

void array_copy(size_t dim, const size_t size[dim], size_t v_size, void* from, void* to)
{
    switch (dim) {
        case 0:
            break;
        case 1:
            memcpy(to, from, size[0] * v_size);
            break;
        default:
            for (size_t i = 0; i < size[0]; i++) {
                array_copy(dim-1, size+1, v_size, ((void**)from)[i], ((void**)to)[i]);
            }
            break;
    }
}

void array_copy_at(size_t dim, size_t v_size, void* from, void* to,
                   const size_t from_index[dim], const size_t to_index[dim])
{
    void* data = array_go_at(dim, v_size, from, from_index);
    array_write_at(dim, v_size, to, data, to_index);
}

void array_set(size_t dim, const size_t size[dim], size_t v_size, void* array, void* data)
{
    void* args[2] = {&v_size, data};
    array_foreach(dim, size, v_size, array, NULL, set_handler, args);
}

void array_read_at(size_t dim, size_t v_size, void* array, void* data, const size_t index[dim])
{
    memcpy(data, array_go_at(dim, v_size, array, index), v_size);
}

void array_write_at(size_t dim, size_t v_size, void* array, void* data, const size_t index[dim])
{
    memcpy(array_go_at(dim, v_size, array, index), data, v_size);
}

void array_roll_axis(size_t dim, const size_t size[dim], size_t v_size, void* array, ssize_t shift, size_t axis)
{
    shift = (size[axis] + shift % size[axis]) % size[axis];
    
    size_t idx_init = 0;
    axis_roll_handler_args args = { size[axis], v_size, array, idx_init, shift, axis };

    do {
        ssize_t fixed_axis[dim];
        memset(fixed_axis, -1, sizeof(fixed_axis));
        fixed_axis[axis] = idx_init;
        
        array_foreach(dim, size, v_size, array, fixed_axis, axis_roll_handler, &args);
        
    } while (++idx_init == (size[axis] + shift + 1) % 2);
}

void array_roll(size_t dim, const size_t size[dim], size_t v_size, void* array, const ssize_t shift[dim])
{
    for (size_t axis = 0; axis < dim; axis++) {
        array_roll_axis(dim, size, v_size, array, shift[axis], axis);
    }
}

void array_roll_axis_to (size_t dim, const size_t size[dim], size_t v_size, void* from, void* to, ssize_t shift, size_t axis)
{
    size_t from_index[dim], to_index[dim];
    memset(from_index, 0, sizeof(from_index));
    
    do {
        memcpy(to_index, from_index, sizeof(to_index));
        to_index[axis] = (from_index[axis] + size[axis] + shift) % size[axis];

        array_copy_at(dim, v_size, from, to, from_index, to_index);
        
    } while ( inc_index(1, dim, size, from_index, -1) == 0 );
}

void array_roll_to(size_t dim, const size_t size[dim], size_t v_size, void* from, void* to, const ssize_t shift[dim])
{
    size_t from_index[dim], to_index[dim];
    memset(from_index, 0, sizeof(from_index));
    
    roll_to_handler_args args = { dim, size, v_size, to, to_index, shift };

    array_foreach(dim, size, v_size, from, NULL, roll_to_handler, &args);
}

void array_foreach(size_t dim, const size_t size[dim], size_t v_size, void* array,
                    ssize_t fixed_axis[dim], array_foreach_callback fct, void* args)
{
    size_t index[dim];
    memset(index, 0, sizeof(index));

    array_iterator(dim, dim, size, v_size, array, index, fixed_axis, fct, args);
}

void array_easy_print(size_t dim, const size_t size[dim], size_t v_size, void* array, array_easy_print_callback print)
{
    easy_print_wrapper_args args = {size, 0, print};
    array_foreach(dim, size, v_size, array, NULL, easy_print_wrapper, &args);
}

void array_set2(size_t size0, size_t size1, size_t v_size, void* array, void* data)
{
    for (size_t x = 0; x < size0; x++) {
        for (size_t y = 0; y < size1; y++) {
            memcpy(&((void***)array)[x][y], data, v_size);
        }
    }
}

void array_roll2_to(size_t size0, size_t size1, size_t v_size, void* from, void* to, const ssize_t shift[2])
{
    for (size_t x = 0; x < size0; x++) {
        size_t x_dst = (x + size0 + shift[0]) % size0;
        for (size_t y = 0; y < size1; y++) {
            size_t y_dst = (y + size1 + shift[1]) % size1;
            memcpy(&((void***)to)[x_dst][y_dst], &((void***)from)[x][y], v_size);
        }
    }
}

void array_set3(size_t size0, size_t size1, size_t size2, size_t v_size, void* array, void* data)
{
    for (size_t x = 0; x < size0; x++) {
        for (size_t y = 0; y < size1; y++) {
            for (size_t z = 0; z < size2; z++) {
                memcpy(&((void****)array)[x][y][z], data, v_size);
            }
        }
    }
}

void array_roll2d_y_s(size_t size0, size_t size1, double array[size0][size1], ssize_t shift)
{
    ssize_t shift_y = (size1 + shift % size1) % size1;
    size_t y_init = 0;
    
    double tmp[2];
    
    do {
        size_t y1, y0 = y_init;
        for (size_t x = 0; x < size0; x++) {
            size_t i = 0;
            tmp[1] = array[x][y0];
            do {
                y1 = (y0 + shift_y) % size1;
                
                tmp[i%2] = array[x][y1];
                array[x][y1] = tmp[(i+1)%2];
                
                i++;
            } while ((y0 = y1) != y_init);
        }
    } while (++y_init == (size1 + shift_y + 1)%2);
}

void array_roll2d_x_s(size_t size0, size_t size1, double array[size0][size1], ssize_t shift)
{
    ssize_t shift_x = (size0 + shift % size0) % size0;
    size_t x_init = 0;
    
    double tmp[2];
    
    do {
        size_t x1, x0 = x_init;
        for (size_t y = 0; y < size1; y++) {
            size_t i = 0;
            tmp[1] = array[x0][y];
            do {
                x1 = (x0 + shift_x) % size0;
                
                tmp[i % 2] = array[x1][y];
                array[x1][y] = tmp[(i+1)%2];
                
                i++;
            } while ((x0 = x1) != x_init);
        }
    } while (++x_init == (size0 + shift_x + 1)%2);
}

void array_roll2d_s(size_t size0, size_t size1, double array[size0][size1], const ssize_t shift[2])
{
    array_roll2d_x_s(size0, size1, array, shift[0]);
    array_roll2d_y_s(size0, size1, array, shift[1]);
}

