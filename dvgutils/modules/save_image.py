import os
import cv2


class SaveImage:

    def __init__(self, output_path, image_ext="jpg", jpg_quality=None, png_compression=None, overwrite=False):
        self.output_path = output_path
        self.image_ext = image_ext
        self.jpg_quality = jpg_quality  # 0 - 100 (higher means better). Default is 95.
        self.png_compression = png_compression  # 0 - 9 (higher means a smaller size and longer compression time). Default is 3.
        self.overwrite = overwrite

    def __call__(self, image, name):
        self.save(image, name)

    def save(self, image, name):
        # Prepare output for image based on name
        output = name.split(os.path.sep)
        dirname = output[:-1]
        if len(dirname) > 0:
            dirname = os.path.join(*dirname)
            dirname = os.path.join(self.output_path, dirname)
        else:
            dirname = self.output_path
        os.makedirs(dirname, exist_ok=True)
        filename = f"{output[-1]}.{self.image_ext}"
        path = os.path.join(dirname, filename)

        if not self.overwrite and os.path.exists(path):
            raise FileExistsError(f"{path} already exists!")

        if self.image_ext == "jpg":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality) if self.jpg_quality else None)
        elif self.image_ext == "png":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_PNG_COMPRESSION, self.png_compression) if self.png_compression else None)
        else:
            raise Exception("Unsupported image extension")
