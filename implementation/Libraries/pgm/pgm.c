/*!
 * \file    pgm.c
 * \brief   PGM image library.
 * \author  Adrien Python
 * \version 2.0 BETA
 * \date    01.12.2011
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "pgm.h"


#define IS_BETWEEN(n,min,max) ((n) >= (min) && (n) <= (max))

#define PGM_TYPE_LEN 2
#define PGM_TYPE "P2"
#define EMPTY_COLOR 255

struct pgm_image {
    size_t width, height;
    unsigned char grayLevel;
    unsigned char **matrix;
};

/*****************************[ Static functions ]*****************************/

/*!
 * \brief   Create a new matrix.
 *
 * \param   width   new matrix width.
 * \param   height  new matrix height.
 * \return  the new matrix.
 */
static unsigned char** pgm_create_matrix(size_t width, size_t height)
{
    unsigned char** matrix = (unsigned char**) malloc(width * sizeof(unsigned char*));
    for (size_t x = 0; x < width; x++) {
        matrix[x] = (unsigned char *) malloc(height * sizeof(unsigned char));
    }
    return matrix;
}

/*!
 * \brief   Destroy a matrix (release its memory).
 *
 * \param   width   matrix width.
 * \param   matrix  matrix to destroy.
 * \return  void.
 */
static void pgm_destroy_matrix(size_t width, unsigned char** matrix)
{
    for (size_t x = 0; x < width; x++) {
        free(matrix[x]);
    }
    free(matrix);
}

/*!
 * \brief   Try to rotate as much pixel as possible (fill their new position).
 *
 * \param   pgm     target pgm_image.
 * \param   alpha   rotation angle.
 * \param   filled  currently filled pixels.
 * \return  void.
 */
static void pgm_partial_rotate(pgm_image* pgm, float alpha, bool** filled)
{
    
    unsigned char** matrix = pgm_create_matrix(pgm->width, pgm->height);
    
    float cosa = cos(alpha);
    float sina = sin(alpha);
    
    /* Rotate the picture */
    for (size_t x = 0; x < pgm->width; x++)
        for (size_t y = 0; y < pgm->height; y++) {
            ssize_t xdiff = x - (int)pgm->width/2;
            ssize_t ydiff = y - (int)pgm->height/2;
            ssize_t x2 = pgm->width/2 + (round(xdiff * cosa - ydiff * sina));
            ssize_t y2 = pgm->height/2 + (round(xdiff * sina + ydiff * cosa));
            if(IS_BETWEEN(x2, 0, (ssize_t)pgm->width-1) && IS_BETWEEN(y2, 0, (ssize_t)pgm->height-1)) {
                matrix[x2][y2] = pgm->matrix[x][y];
                filled[x2][y2] = true;
            }
        }
    
    pgm_destroy_matrix(pgm->width, pgm->matrix);
    pgm->matrix = matrix;
}

/*!
 * \brief   "Heal" (fill) an empty pixel which have at least 2 neighbors with
 *          their average color.
 *
 * \param   pgm     target pgm_image.
 * \param   filled  currently filled pixels.
 * \param   x       x coordinate of the pixel to heal.
 * \param   y       y coordinate of the pixel to heal.
 * \return  void.
 */
static void pgm_heal_pixel(pgm_image* pgm, bool** filled, size_t x, size_t y)
{
    int pixCnt = 0;
    int colorSum = 0;
    for (ssize_t x2 = x-1; x2 <= (ssize_t)x+1; x2++) {
        for (ssize_t y2 = y-1; y2 <= (ssize_t)y+1; y2++) {
            if(IS_BETWEEN(x2, 0, (ssize_t)pgm->width-1) && IS_BETWEEN(y2, 0, (ssize_t)pgm->height-1)) {
                if(filled[x2][y2]) {
                    pixCnt++;
                    colorSum += pgm->matrix[x2][y2];
                }
            }
        }
    }
    if (pixCnt > 1) {
        pgm->matrix[x][y] = colorSum / pixCnt;
        filled[x][y] = true;
    }
}

/*!
 * \brief   "Heal" (fill) the empty pixels of a partially rotated picture.
 *
 * \param   pgm     target pgm_image.
 * \param   filled  currently filled pixels.
 * \return  void.
 */
static void pgm_heal_partial_rotation(pgm_image* pgm, bool** filled)
{
    for (size_t x = 0; x < pgm->width; x++) {
        for (size_t y = 0; y < pgm->height; y++) {
            if( ! filled[x][y] ) {
                pgm_heal_pixel(pgm, filled, x, y);
            }
        }
    }
}

/*!
 * \brief   Fill the remaining empty pixel.
 *
 * \param   pgm     target pgm_image.
 * \param   filled  currently filled pixels.
 * \return  void.
 */
static void pgm_fill_holes(pgm_image* pgm, bool **filled)
{
    for (size_t x = 0; x < pgm->width; x++) {
        for (size_t y = 0; y < pgm->height; y++) {
            if( ! filled[x][y] ) {
                pgm->matrix[x][y] = EMPTY_COLOR;
            }
        }
    }
}

/*****************************[ Public functions ]*****************************/

pgm_image* pgm_load(char* filename)
{
    size_t len = 0;
    char type[PGM_TYPE_LEN], *comment = NULL;
    
    FILE* fid = fopen(filename, "r");
	
    if ( fid == NULL ) goto open_file_error;
    
    pgm_image* pgm = malloc(sizeof(pgm_image));

    if ( fread(type, sizeof(char), PGM_TYPE_LEN, fid) != PGM_TYPE_LEN)    goto load_type_error;
    if ( strncmp(PGM_TYPE, type, PGM_TYPE_LEN) != 0)                      goto file_type_error;
    if ( getline(&comment, &len, fid) == -1)        goto load_comment_error;
    if ( fscanf(fid, "%lu", &pgm->width) != 1)      goto load_size_error;
    if ( fscanf(fid, "%lu", &pgm->height) != 1)     goto load_size_error;
    if ( fscanf(fid, "%hhu", &pgm->grayLevel) != 1) goto load_graylevel_error;
    
    pgm->matrix = pgm_create_matrix(pgm->width, pgm->height);
    for (size_t y = 0; y < pgm->height; y++) {
        for (size_t x = 0; x < pgm->width; x++) {
            if( fscanf(fid, "%hhu", &(pgm->matrix[x][y])) != 1) {
                goto load_matrix_error;
            }
        }
    }
    
    free(comment);
    
    fclose(fid);
    
    return pgm;

    /* Errors handling */
    
load_matrix_error:
    fclose(fid);
    pgm_destroy_matrix(pgm->width, pgm->matrix);
load_graylevel_error:
load_size_error:
    free(comment);
load_comment_error:
file_type_error:
load_type_error:
    free(pgm);
open_file_error:
    
    return NULL;
}

pgm_image* pgm_create(size_t width, size_t height)
{
    
    pgm_image* pgm = NULL;
    
    pgm = malloc(sizeof(pgm_image));
    
    pgm->width = width;
    pgm->height = height;
    pgm->grayLevel = 255;
    
    pgm->matrix = pgm_create_matrix(pgm->width, pgm->height);
    for (size_t x = 0; x < pgm->width; x++)
        for (size_t y = 0; y < pgm->height; y++)
            pgm->matrix[x][y] = 0;

    return pgm;
}

void pgm_destroy(pgm_image* pgm)
{
    pgm_destroy_matrix(pgm->width, pgm->matrix);
    free(pgm);
}

void pgm_set_pixel(pgm_image* pgm, size_t x, size_t y, int color) {
    pgm->matrix[x][y] = color;
}

void pgm_write(pgm_image* pgm, char* fileName)
{
	FILE* fid = fopen(fileName, "w");
	
	fprintf(fid, "%s\n", PGM_TYPE);
	fprintf(fid, "# CREATOR: PGMLIB 2.0 BETA\n");
	fprintf(fid, "%lu %lu\n", pgm->width, pgm->height);
	fprintf(fid, "%d\n", pgm->grayLevel);
	
    for (size_t y = 0; y < pgm->height; y++) {
        for (size_t x = 0; x < pgm->width; x++) {
            fprintf(fid, "%hhu ", pgm->matrix[x][y]);
        }
        fprintf(fid, "\n");
    }
	
	fclose(fid);
}

void pgm_reverse_color(pgm_image* pgm)
{
	for (size_t x = 0; x < pgm->width; x++)
		for (size_t y = 0; y < pgm->height; y++)
			pgm->matrix[x][y] = pgm->grayLevel - pgm->matrix[x][y];
}

void pgm_vertical_symmetry(pgm_image* pgm)
{
    unsigned char** matrix = pgm_create_matrix(pgm->width, pgm->height);
	
	for (size_t x = 0; x < pgm->width; x++)
		for (size_t y = 0; y < pgm->height; y++) {
			matrix[x][y] = pgm->matrix[pgm->width - x - 1][y];
		}
    pgm_destroy_matrix(pgm->width, pgm->matrix);
    pgm->matrix = matrix;
}

void pgm_horizontal_symmetry(pgm_image* pgm)
{
	
    unsigned char** matrix = pgm_create_matrix(pgm->width, pgm->height);
	
	for (size_t x = 0; x < pgm->width; x++)
		for (size_t y = 0; y < pgm->height; y++) {
			matrix[x][y] = pgm->matrix[x][pgm->width - y - 1];
		}
    pgm_destroy_matrix(pgm->width, pgm->matrix);
	pgm->matrix = matrix;
}

void pgm_photo_booth(pgm_image* pgm)
{
    unsigned char** matrix = pgm_create_matrix(pgm->width, pgm->height);
	
	int xShift = (int)pgm->width/2 + pgm->width%2;
	int yShift = (int)pgm->height/2 + pgm->height%2;
	
	for (size_t x = 0; x < pgm->width; x++)
		for (size_t y = 0; y < pgm->height; y++) {
			/* Evens pixels are written on the first picture, Odds are shifted*/
			size_t x2 = (x/2) + (x%2) * xShift;
			size_t y2 = (y/2) + (y%2) * yShift;
			matrix[x2][y2] = pgm->matrix[x][y];
		}
    pgm_destroy_matrix(pgm->width, pgm->matrix);
    pgm->matrix = matrix;
}

void pgm_rotate(pgm_image* pgm, float alpha)
{
    // Matrix of filled pixels (initialize at false)
    bool **filled = (bool**) malloc(pgm->width * sizeof(bool*));
    for (size_t x = 0; x < pgm->width; x++) {
        filled[x] = (bool*) calloc(pgm->height, sizeof(bool));
    }
    
    pgm_partial_rotate(pgm, alpha, filled);
    pgm_heal_partial_rotation(pgm, filled);
    pgm_fill_holes(pgm, filled);
    
    for (size_t x = 0; x < pgm->width; x++) {
        free(filled[x]);
    }
    free(filled);
}

void pgm_basic_zoom(pgm_image* pgm, int zoom)
{

    size_t newWidth = pgm->width * zoom;
    size_t newHeight = pgm->height * zoom;
    
    unsigned char** matrix = pgm_create_matrix(newWidth, newHeight);
    
	for (size_t x = 0; x < pgm->width; x++)
		for (size_t y = 0; y < pgm->height; y++)
			// Replicate the current pixel in the new and bigger one
            for (size_t x2 = x * zoom; x2 < (x+1) * zoom; x2++)
                for (size_t y2 = y * zoom; y2 < (y+1)*zoom; y2++){
					matrix[x2][y2] = pgm->matrix[x][y];
                }

    pgm_destroy_matrix(pgm->width, pgm->matrix);
    pgm->matrix = matrix;
    pgm->width = newWidth;
    pgm->height = newHeight;
}

void pgm_set_matrix(pgm_image* pgm, size_t width, size_t height, unsigned char** matrix)
{
    pgm_destroy_matrix(pgm->width, pgm->matrix);
    pgm->width = width;
    pgm->height = height;
    pgm->matrix = matrix;
}

void pgm_resize(pgm_image* pgm, size_t width, size_t height)
{
    size_t w = width;
    
    if (pgm->width != width) {
        
        size_t new_matrix_size = sizeof(unsigned char*) * width;
        
        if (width > pgm->width) {
            // Add missing columns
            pgm->matrix = realloc(pgm->matrix, new_matrix_size);
            for (size_t x = pgm->width; x < width; x++) {
                size_t y_size = sizeof(unsigned char) * height;
                pgm->matrix[x] = malloc(y_size);
                memset(pgm->matrix[x], EMPTY_COLOR, y_size);
            }
            w = pgm->width;
        } else {
            // Remove additional columns
            for (size_t x = pgm->width; x < width; x++) {
                free(pgm->matrix[x]);
            }
            pgm->matrix = realloc(pgm->matrix, new_matrix_size);
            w = width;
        }
        pgm->width = width;
    }

    if (pgm->height != height) {
        for (size_t x = 0; x < w; x++) {
            // Expend/reduce colomn size
            pgm->matrix[x] = realloc(pgm->matrix[x], sizeof(unsigned char) * height);
            if (height > pgm->height) {
                // fill the new space
                size_t fill_space = sizeof(unsigned char) * (height - pgm->height);
                memset(&(pgm->matrix[x][pgm->height]), EMPTY_COLOR, fill_space);
            }
        }
        pgm->height = height;
    }
}

void pgm_get_info(pgm_image* pgm, pgm_info* info)
{
    info->width = pgm->width;
    info->height = pgm->height;
    info->grayLevel = pgm->grayLevel;
}

