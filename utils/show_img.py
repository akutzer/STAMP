
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None


def check_tiff(path):
    try:
        with Image.open(path) as img:
            img.verify()  # verify TIFF files
        print(f"{path} - OK")
    except Exception as e:
        print(f"{path} - Error: {e}")


def display_image(image_path):
    with Image.open(image_path) as img:
        plt.imshow(img)
        plt.axis('off')
        plt.show()


# check the image or mask 
check_tiff('/mnt/bulk-io/maurice/LEOPARD_CHALLENGE/data/wsi/case_radboud_0626.tif')
#check_tiff('/mnt/bulk-io/maurice/LEOPARD_CHALLENGE/data/leopard_train_tissue_masks/case_radboud_0626_tissue.tif')


# show the image or mask 
display_image('/mnt/bulk-io/maurice/LEOPARD_CHALLENGE/data/wsi/case_radboud_0626.tif')
#display_image('/mnt/bulk-io/maurice/LEOPARD_CHALLENGE/data/leopard_train_tissue_masks/case_radboud_0626_tissue.tif')

