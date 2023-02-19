__device__
void SetPixel(uint8_t* image, int width, int xx, int yy, uint8_t red, uint8_t grn, uint8_t blu)
{
    uint8_t* pixel = &image[(yy*width+xx)*4];
    pixel[1] = red;
    pixel[2] = grn;
    pixel[3] = blu;
    pixel[0] = 255;  // no alpha
    //printf("%i %i %i\n", red, blu, grn);
}

__device__
void SetPixelFloat(uint8_t* image, int width, int xx, int yy, float fred, float fgrn, float fblu )
{
    // convert float to unorm
    uint8_t red = (uint8_t)roundf( 255.0f * fred );
    uint8_t grn = (uint8_t)roundf( 255.0f * fgrn );
    uint8_t blu = (uint8_t)roundf( 255.0f * fblu );
    
    SetPixel(image, width, xx, yy, red, grn, blu);
}

