import sys, argparse, logging
import lib_fits
from lib_fits import AllImages
import glob


parser = argparse.ArgumentParser(description='Convolve images to common beam, e.g. beam_convolve.py  --noise --sigma 5 --save *fits')
parser.add_argument('basename', type=str, help='basename and path of all images to convolve beam for')
parser.add_argument('--beam', dest='beam', nargs='+', type=float, help='3 parameters final beam to convolve all images (BMAJ (arcsec), BMIN (arcsec), BPA (deg)), if none convolve to smalles common beam')
parser.add_argument('--bgreg', dest='bgreg', help='DS9 region file for background estimation.')
parser.add_argument('--noise', dest='noise', action='store_true', help='Calculate noise of each image')
parser.add_argument('--sigma', dest='sigma', type=float, help='Restrict to pixels above this sigma in all images')
parser.add_argument('--circbeam', dest='circbeam', action='store_true', help='Force final beam to be circular (default: False, use minimum common beam area)')
parser.add_argument('--save', dest='save', action='store_true', help='Save results')
parser.add_argument('--force', dest='force', action='store_true', help='Force reconvolve images')
args = parser.parse_args()

# check input

if args.beam is not None and len(args.beam) != 3:
    print('Beam must be in the form of "BMAJ BMIN BPA" (3 floats).')
    sys.exit(1)

if args.sigma and not args.noise:
    print('Cannot use --sigma flag without calculating noise. Provide also --noise.')
    sys.exit(1)



########################################################
# prepare images and make catalogues if necessary
basename = args.basename
images=glob.glob(basename+'0*image.fits')

all_images = AllImages([imagefile for imagefile in images])

#########################################################
# convolve
if all_images.suffix_exists('-conv') and not args.force:
    print('Reuse -conv images.')
    all_images = lib_fits.AllImages([name.replace('.fits', '-conv.fits') for name in all_images.filenames])
else:
    if args.beam:
        print(f"Convolving to specified beam: {args.beam}")
        all_images.convolve_to(args.beam, args.circbeam)
    else:
        print(f"Convolving to smallest common beam")
        all_images.convolve_to(circbeam=args.circbeam)
    if args.save: 
        all_images.write('-conv')

for i, image in enumerate(all_images):
    if args.noise:
        if args.sigma:
            image.calc_noise(sigma=args.sigma, bg_reg=args.bgreg)  # after mask?/convolution
            logging.info(f"Noise for {image.filename}: {image.noise}")
            print(f"Noise for {image.filename}: {image.noise}")
            image.blank_noisy(args.sigma)
        else:
            image.calc_noise() # after mask?/convolution
            logging.info(f"Noise for {image.filename}: {image.noise}")
            print(f"Noise for {image.filename}: {image.noise}")