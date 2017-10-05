/*!
 * \file    array.h
 * \brief   Array toolbox
 * \author  Adrien Python
 * \version 1.0
 * \date    21.11.2016
 * \warning Functions designed for array of generic dimension are slower than
 *          those designed for a specific dimension.
 */

#ifndef ARRAY_H
#define ARRAY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/***************** Functions for arrays of a generic dimension ****************/

/**
 * \brief Callback function type for array_foreach().
 *
 * Function automatically called foreach cell of the target array.
 *
 * \param   dim    array dimension
 * \param   data   pointer to the cell data
 * \param   index  current cell index
 * \param   args   callback function additional arguments
 * \return  indicates whether array_foreach should continue to iterate or not.
 */
typedef bool (*array_foreach_callback)(size_t dim, void* data, const size_t* index, void* args);

/**
 * \brief Callback function type for array_easy_print().
 *
 * Function automatically called foreach cell to print.
 *
 * \param   data  pointer to the cell data
 * \return  void
 */
typedef void (*array_easy_print_callback)(void* data);

/*!
 * \brief  Create an N dimensional array.
 *
 * Create an N dimensional array like N nested loops with somee malloc() would.
 *
 * The created array should be freed by array_destroy, or manually (e.g. through
 * some free() in N nested loops).
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  data size (e.g. sizeof(int) for an array of int)
 * \return  created array
 */
void* array_create(size_t dim, const size_t* size, size_t v_size);

/*!
 * \brief  Destroy an N dimensional array.
 *
 * Destroy an N dimensional array like N nested loops with some free() would.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  data size (e.g. sizeof(int) for an int array)
 * \param   array   array to free
 * \return  void
 */
void array_destroy(size_t dim, const size_t* size, size_t v_size, void* array);

/*!
 * \brief  Build an N dimensional array mapping on a chunk of memory.
 *
 * Build a mapping on a static or dynamic memory as an N dimensional array.
 *
 * \param   dim     mapping dimension
 * \param   size    mapping dimensions sizes
 * \param   v_size  mapping data size (e.g. sizeof(int) for int array mapping)
 * \param   array   chunk of memory to map (base array)
 * \return  created mapping
 *
 * Example Usage:
 * \code
 *    int32_t array[100]; // 400 bytes available for mapping
 *
 *    // 10x10 2D array mapping
 *    size_t map2d_size[2] = {10, 10};
 *    int32_t**  map2d = array_map(2, map2d_size, sizeof(int32_t), array);
 *
 *    // 4x10x10 3D array mapping
 *    size_t map3d_size[3] = {4, 10, 10};
 *    int_8_t*** map3d = array_map(3, map3d_size, sizeof(int8_t), array);
 * \endcode
 *
 */
void* array_map(size_t dim, const size_t* size, size_t v_size, void* array);

/*!
 * \brief  Destroy an N dimensional array mapping.
 *
 * Dynamically allocated mapped chunk of memory (base array) is not destroyed.
 *
 * \param   dim      mapping dimension
 * \param   size     mapping dimensions sizes
 * \param   v_size   mapping data size (e.g. sizeof(int) for int array mapping)
 * \param   mapping  mapping to free
 * \return  void
 */
void array_unmap(size_t dim, const size_t* size, size_t v_size, void* mapping);

/*!
 * \brief  Copy an array to an other of the same dimension and sizes.
 *
 * \param   dim      array dimension
 * \param   size     array dimensions sizes
 * \param   v_size   array data size (e.g. sizeof(int) for int array)
 * \param   from     source array
 * \param   to       destination array
 * \return  void
 */
void array_copy(size_t dim, const size_t* size, size_t v_size, void* from, void* to);

/*!
 * \brief  Copy the content of an array to another from and at specific indexes.
 *
 * \param   dim         array dimension
 * \param   v_size      array data size (e.g. sizeof(int) for int array)
 * \param   from        source array
 * \param   to          destination array
 * \param   from_index  source index (where's the data to copy)
 * \param   to_index    destination index (where's the copied data)
 * \return  void
 */
void array_copy_at(size_t dim, size_t v_size, void* from, void* to,
                   const size_t* from_index, const size_t* to_index);

/*!
 * \brief  Set the content of an array.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to set
 * \param   data    data to write in the array
 * \return  void
 */
void array_set(size_t dim, const size_t* size, size_t v_size, void* array, void* data);

/*!
 * \brief  Read the content of an array at the specified index.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to set
 * \param   data    storing location for the read data
 * \param   index   position where to read
 * \return  void
 */
void array_read_at(size_t dim, size_t v_size, void* array, void* data, const size_t* index);

/*!
 * \brief  Write the content of an array at the specified index.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   array to write on
 * \param   data    data to be write in the array
 * \param   index   position where to write
 * \return  void
 */
void array_write_at(size_t dim, size_t v_size, void* array, void* data, const size_t* index);

/*!
 * \brief  Roll an axis of an array (in place).
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to roll
 * \param   shift   shift applied to the roll
 * \param   axis    axis to roll
 * \return  void
 */
void array_roll_axis(size_t dim, const size_t* size, size_t v_size, void* array, ssize_t shift, size_t axis);

/*!
 * \brief  Roll an entire array (in place).
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to roll
 * \param   shift   shift applied to the roll for each axis
 * \return  void
 */
void array_roll(size_t dim, const size_t* size, size_t v_size, void* array, const ssize_t* shift);

/*!
 * \brief  Roll an axis of an array and store the result in another.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   from    arret to roll
 * \param   in      result of the roll
 * \param   shift   shift applied to the roll
 * \param   axis    axis to roll
 * \return  void
 */
void array_roll_axis_to (size_t dim, const size_t* size, size_t v_size, void* from, void* to, ssize_t shift, size_t axis);

/*!
 * \brief  Roll an entire array and store the result in another.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   from    arret to roll
 * \param   in      result of the roll
 * \param   shift   shift applied to the roll
 * \return  void
 */
void array_roll_to(size_t dim, const size_t* size, size_t v_size, void* from, void* to, const ssize_t* shift);

/*!
 * \brief  Call a callback function for each array cells.
 *
 * \param   dim         array dimension
 * \param   size        array dimensions sizes
 * \param   v_size      array data size (e.g. sizeof(int) for int array)
 * \param   array       arret to iterate
 * \param   fixed_axis  axises to set to a fixed index; so, with dim=3:
 *                       -# {-1, -1, -1}: iterate all {x,y,z} array indexes
 *                       -# {-1, 4, -1}: iterate array for all {x, 4, z} indexes
 *                       -# NULL: equivalent to {-1, -1, -1}
 * \param   callback    function to call for each array cell
 * \param   args        callback function additional arguments
 * \return  void
 */
void array_foreach(size_t dim, const size_t* size, size_t v_size, void* array, ssize_t* fixed_axis, array_foreach_callback callback, void* args);

/*!
 * \brief  Easy priniting function for any array.
 *
 * \param   dim     array dimension
 * \param   size    array dimensions sizes
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to print
 * \param   print   function knowing how to print the array data
 * \return  void
 */
void array_easy_print(size_t dim, const size_t* size, size_t v_size, void* array, array_easy_print_callback print);

/**************** Functions for arrays of a specific dimension ****************/

/*!
 * \brief  Set the content of a two-dimensional array.
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to set
 * \param   data    data to write in the array
 * \return  void
 */
void array_set2(size_t size0, size_t size1, size_t v_size, void* array, void* data);

/*!
 * \brief  Roll an entire two-dimensional array and store the result in another.
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   from    arret to roll
 * \param   in      result of the roll
 * \param   shift   shift applied to the roll
 * \return  void
 */
void array_roll2_to(size_t size0, size_t size1, size_t v_size, void* from, void* to, const ssize_t shift[2]);

/*!
 * \brief  Set the content of a three-dimensional array.
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   size2   size of the thrid dimension
 * \param   v_size  array data size (e.g. sizeof(int) for int array)
 * \param   array   arret to set
 * \param   data    data to write in the array
 * \return  void
 */
void array_set3(size_t size0, size_t size1, size_t size2, size_t v_size, void* array, void* data);

#ifdef ARRAY_STATIC_FUNCS

/*!
 * \brief  Roll the y axis of a two-dimensional double static array (in place).
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   array   arret to roll
 * \param   shift   shift applied to the roll
 * \return  void
 */
void array_roll2d_y_s(size_t size0, size_t size1, double array[size0][size1], ssize_t shift);

/*!
 * \brief  Roll the x axis of a two-dimensional double static array (in place).
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   array   arret to roll
 * \param   shift   shift applied to the roll
 * \return  void
 */
void array_roll2d_x_s(size_t size0, size_t size1, double array[size0][size1], ssize_t shift);

/*!
 * \brief  Roll an entire two-dimensional double static array (in place).
 *
 * \param   size0   size of the first dimension
 * \param   size1   size of the second dimension
 * \param   array   arret to roll
 * \param   shift   shift applied to the roll of axis
 * \return  void
 */
void array_roll2d_s(size_t size0, size_t size1, double array[size0][size1], const ssize_t shift[2]);

#endif /* ARRAY_STATIC_FUNCS */
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* ARRAY_H */
