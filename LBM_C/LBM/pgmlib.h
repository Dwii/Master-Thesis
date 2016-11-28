/*!
 * \file    pgmlib.h
 * \brief   PGM image library.
 * \author  Adrien Python
 * \version 2.0 BETA
 * \date    01.12.2011
 */

#ifndef __PGMLIB_H
#define __PGMLIB_H

typedef struct pgm_image pgm_image;

typedef struct {
    size_t width, height;
    unsigned char grayLevel;
} pgm_info;

/*!
 * \brief  Load a PGM file in a pgm_image.
 *
 * \param   filename  file to load.
 * \return  loaded    PGM data.
 */
pgm_image* pgm_load(char* filename);

/*!
 * \brief   Create a pgm_image from scratch.
 *
 * \param   width   width of the PGM picture.
 * \param   height  height of the PGM picture.
 * \return  created PGM data.
 */
pgm_image* pgm_create(size_t width, size_t height);

/*!
 * \brief   Write a pgm_image in a PGM file.
 *
 * \param   pgm       pgm_image to write in a file.
 * \param   filename  file to write.
 * \return  void
 */
void pgm_write(pgm_image* pgm, char* filename);

/*!
 * \brief   Destroy a pgm_image (release its memory).
 *
 * \param   pgm   pgm_image to destroy.
 * \return  void
 */
void pgm_destroy(pgm_image* pgm);

/*!
 * \brief   Reverse the colors of a pgm_image.
 *
 * \param   pgm  target pgm_image.
 * \return  void
 */
void pgm_reverse_color(pgm_image* pgm);

/*!
 * \brief   Apply a vertical symmetry on a pgm_image.
 *
 * \param   pgm  target pgm_image.
 * \return  void
 */
void pgm_vertical_symmetry(pgm_image* pgm);

/*!
 * \brief   Apply an horizontal symmetry on a pgm_image.
 *
 * \param   pgm  target pgm_image.
 * \return  void
 */
void pgm_horizontal_symmetry(pgm_image* pgm);

/*!
 * \brief   Apply a lossless photo booth effect on a pgm_image.
 *
 * \param   pgm  target pgm_image.
 * \return  void
 */
void pgm_photo_booth(pgm_image* pgm);

/*!
 * \brief   Rotate a pgm_image.
 *
 * \todo    Resize the picture to display the full rotation.
 *
 * \param   pgm    target pgm_image.
 * \param   alpha  rotation angle.
 * \return  void
 */
void pgm_rotate(pgm_image* pgm, float alpha);

/*!
 * \brief   Zoom on a pgm_image.
 *
 * \param   pgm   target pgm_image.
 * \param   zoom  zoom multiplier.
 * \return  void
 */
void pgm_basic_zoom(pgm_image* pgm, int zoom);

/*!
 * \brief   Set a single pixel color of a pgm_image.
 *
 * \param   pgm    target pgm_image.
 * \param   x      pixel x coordinate.
 * \param   y      pixel y coordinate.
 * \param   color  now pixel color.
 * \return  void
 */
void pgm_set_pixel(pgm_image* pgm, size_t x, size_t y, int color);

/*!
 * \brief   Set the pgm_picture matrix.
 *
 * \param   pgm     target pgm_image.
 * \param   width   new matrix width.
 * \param   height  new matrix height.
 * \param   matrix  new matrix.
 * \return  void
 */
void pgm_set_matrix(pgm_image* pgm, size_t width, size_t height, unsigned char** matrix);

/*!
 * \brief   Resize a pgm_picture matrix.
 *
 * \param   pgm     target pgm_image.
 * \param   width   new image width.
 * \param   height  new image height.
 * \return  void
 */
void pgm_resize(pgm_image* pgm, size_t width, size_t height);

/*!
 * \brief   Get the pgm_picture informations.
 *
 * \param   pgm        target pgm_image.
 * \param   info[out]  information to load.
 * \return  void
 */
void pgm_get_info(pgm_image* pgm, pgm_info* info);


#endif  /* __PGMLIB_H */
